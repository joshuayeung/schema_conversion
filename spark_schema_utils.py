from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, NullType
from pyspark.sql.functions import lit, col, when, isnull, array, struct, map_from_arrays, transform, expr

def add_missing_columns(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Adds missing columns and nested subfields to a PySpark DataFrame based on the provided schema.
    Missing columns and subfields (including nested structs, maps, arrays) are added as NULL.
    
    Args:
        df: Input PySpark DataFrame
        schema: Target schema (StructType) to align the DataFrame with
    
    Returns:
        DataFrame with all columns and nested subfields from the schema, missing ones added as NULL
    """
    def get_field_expr(field: StructField, prefix: str = "", existing_field: StructField = None, is_array_element: bool = False, lambda_var: str = None) -> str:
        """Generate expression for a field, handling nested structures and existing fields."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix and not is_array_element else field.name
        
        if isinstance(field_type, StructType):
            existing_struct_type = existing_field.dataType if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, StructType) else None
            existing_subfields = {f.name: f for f in existing_struct_type.fields} if existing_struct_type else {}
            
            subfield_exprs = []
            for subfield in field_type.fields:
                existing_subfield = existing_subfields.get(subfield.name)
                subfield_prefix = f"{field_name}." if field_name and not is_array_element else ""
                subfield_lambda_var = f"{lambda_var}.{subfield.name}" if is_array_element and lambda_var else None
                subfield_expr = get_field_expr(
                    subfield,
                    subfield_prefix,
                    existing_subfield,
                    is_array_element=is_array_element,
                    lambda_var=subfield_lambda_var
                )
                subfield_expr = subfield_expr.alias(subfield.name)
                subfield_exprs.append(subfield_expr)
            
            return struct(*subfield_exprs).alias(field_name) if not is_array_element else struct(*subfield_exprs)
                
        elif isinstance(field_type, ArrayType):
            if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, ArrayType):
                if isinstance(field_type.elementType, (StructType, ArrayType)):
                    existing_element_field = StructField(field.name, existing_field.dataType.elementType, True) if isinstance(existing_field.dataType.elementType, (StructType, ArrayType)) else None
                    element_expr = get_field_expr(
                        StructField(field.name, field_type.elementType, True),
                        prefix="",
                        existing_field=existing_element_field,
                        is_array_element=True,
                        lambda_var="x"
                    )
                    return when(
                        col(field_name).isNotNull(),
                        transform(col(field_name), lambda x: element_expr)
                    ).otherwise(
                        array().cast(field_type)
                    ).alias(field_name)
                return col(field_name)
            return array().cast(field_type).alias(field_name)
            
        elif isinstance(field_type, MapType):
            if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, MapType):
                if isinstance(field_type.valueType, (StructType, MapType)):
                    existing_value_field = StructField(field.name, existing_field.dataType.valueType, True) if isinstance(existing_field.dataType.valueType, (StructType, MapType)) else None
                    value_expr = get_field_expr(
                        StructField(field.name, field_type.valueType, True),
                        prefix="",
                        existing_field=existing_value_field,
                        is_array_element=True,
                        lambda_var="v"
                    )
                    return when(
                        col(field_name).isNotNull(),
                        expr(f"transform_values({field_name}, (k, v) -> {value_expr})")
                    ).otherwise(
                        map_from_arrays(array(), array()).cast(field_type)
                    ).alias(field_name)
                return col(field_name)
            return map_from_arrays(array(), array()).cast(field_type).alias(field_name)
            
        else:
            if is_array_element and lambda_var:
                return col(lambda_var)
            if isinstance(existing_field, StructField):
                return col(field_name)
            return lit(None).cast(field_type).alias(field_name)
    
    existing_columns = {f.name: f for f in df.schema.fields}
    select_expr = []
    
    for field in schema.fields:
        existing_field = existing_columns.get(field.name)
        select_expr.append(get_field_expr(field, "", existing_field))
    
    return df.select(*select_expr)

def normalize_nulls(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Normalizes nullability in a PySpark DataFrame based on the schema.
    For optional structs/maps/arrays with required nested fields, sets the entire
    field to NULL if all required nested fields are NULL.
    
    Args:
        df: Input PySpark DataFrame
        schema: Schema (StructType) defining nullability constraints
    
    Returns:
        DataFrame with normalized NULL values for nested structures
    """
    def build_null_condition(field: StructField, prefix: str = "", is_array_element: bool = False, lambda_var: str = None) -> str:
        """Build condition to check if a field should be set to NULL."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix and not is_array_element else field.name
        
        if isinstance(field_type, StructType):
            required_fields = [f for f in field_type.fields if not f.nullable]
            if not required_fields:
                return col(field_name) if not is_array_element else col(lambda_var)
            
            subfields = [
                build_null_condition(
                    subfield,
                    f"{field_name}." if not is_array_element else "",
                    is_array_element=is_array_element,
                    lambda_var=f"{lambda_var}.{subfield.name}" if is_array_element and lambda_var else None
                ).alias(subfield.name)
                for subfield in field_type.fields
            ]
            
            if field.nullable:
                conditions = [
                    col(f"{field_name}.{f.name}").isNull() if not is_array_element else col(f"{lambda_var}.{f.name}").isNull()
                    for f in required_fields
                ]
                combined_condition = conditions[0]
                for cond in conditions[1:]:
                    combined_condition = combined_condition & cond
                
                return when(
                    combined_condition,
                    lit(None)
                ).otherwise(
                    struct(*subfields)
                ).alias(field_name) if not is_array_element else when(
                    combined_condition,
                    lit(None)
                ).otherwise(
                    struct(*subfields)
                )
            else:
                return struct(*subfields).alias(field_name) if not is_array_element else struct(*subfields)
                
        elif isinstance(field_type, ArrayType):
            if field.nullable and isinstance(field_type.elementType, (StructType, ArrayType)):
                element_condition = build_null_condition(
                    StructField("element", field_type.elementType, True),
                    prefix="",
                    is_array_element=True,
                    lambda_var="x"
                )
                return when(
                    col(field_name).isNull(),
                    lit(None)
                ).otherwise(
                    transform(col(field_name), lambda x: element_condition).cast(field_type)
                ).alias(field_name)
            return col(field_name)
            
        elif isinstance(field_type, MapType):
            if field.nullable and isinstance(field_type.valueType, StructType):
                value_condition = build_null_condition(
                    StructField("value", field_type.valueType, True),
                    prefix="",
                    is_array_element=True,
                    lambda_var="v"
                )
                return when(
                    col(field_name).isNull(),
                    lit(None)
                ).otherwise(
                    expr(f"transform_values({field_name}, (k, v) -> {value_condition})").cast(field_type)
                ).alias(field_name)
            return col(field_name)
            
        else:
            return col(field_name) if not is_array_element else col(lambda_var)
    
    select_expr = [build_null_condition(field) for field in schema.fields]
    
    return df.select(*select_expr)
