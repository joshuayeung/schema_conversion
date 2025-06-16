from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, NullType
from pyspark.sql.functions import lit, col, when, array, struct, map_from_arrays, coalesce, expr

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
    def get_field_expr(field: StructField, prefix: str = "", existing_field: StructField = None) -> str:
        """Generate expression for a field, handling nested structures and existing fields."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix else field.name
        
        if isinstance(field_type, StructType):
            existing_struct_type = existing_field.dataType if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, StructType) else None
            existing_subfields = {f.name: f for f in existing_struct_type.fields} if existing_struct_type else {}
            
            subfield_exprs = []
            for subfield in field_type.fields:
                existing_subfield = existing_subfields.get(subfield.name)
                if existing_subfield and isinstance(existing_field, StructField):
                    subfield_expr = col(f"{field_name}.{subfield.name}")
                    if isinstance(subfield.dataType, (StructType, ArrayType, MapType)):
                        nested_expr = get_field_expr(
                            subfield,
                            f"{field_name}." if field_name else "",
                            existing_subfield
                        )
                        subfield_expr = nested_expr
                else:
                    subfield_expr = get_field_expr(
                        subfield,
                        f"{field_name}." if field_name else "",
                        None
                    )
                subfield_expr = subfield_expr.alias(subfield.name)
                subfield_exprs.append(subfield_expr)
            
            return struct(*subfield_exprs).alias(field_name)
                
        elif isinstance(field_type, ArrayType):
            if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, ArrayType):
                return col(field_name).alias(field_name)
            return array().cast(field_type).alias(field_name)
            
        elif isinstance(field_type, MapType):
            if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, MapType):
                return col(field_name).alias(field_name)
            return map_from_arrays(array(), array()).cast(field_type).alias(field_name)
            
        else:
            if isinstance(existing_field, StructField):
                return col(field_name).alias(field_name)
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
    def build_null_condition(field: StructField, prefix: str = "", existing_field: StructField = None) -> str:
        """Build condition to check if a field should be set to NULL."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix else field.name
        
        if isinstance(field_type, StructType):
            existing_struct_type = existing_field.dataType if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, StructType) else None
            existing_subfields = {f.name: f for f in existing_struct_type.fields} if existing_struct_type else {}
            
            required_fields = [f for f in field_type.fields if not f.nullable]
            if not required_fields:
                return col(field_name).alias(field_name)
            
            subfield_exprs = []
            conditions = []
            for subfield in field_type.fields:
                existing_subfield = existing_subfields.get(subfield.name)
                subfield_expr = build_null_condition(
                    subfield,
                    f"{field_name}." if field_name else "",
                    existing_subfield
                )
                subfield_expr = subfield_expr.alias(subfield.name)
                subfield_exprs.append(subfield_expr)
                if not subfield.nullable:
                    conditions.append(col(f"{field_name}.{subfield.name}").isNull())
            
            combined_condition = conditions[0] if conditions else lit(True)
            for cond in conditions[1:]:
                combined_condition = combined_condition & cond
            
            if field.nullable:
                return when(
                    combined_condition,
                    lit(None)
                ).otherwise(
                    struct(*subfield_exprs)
                ).alias(field_name)
            return struct(*subfield_exprs).alias(field_name)
                
        elif isinstance(field_type, ArrayType):
            if field.nullable:
                return when(
                    col(field_name).isNull() | (size(col(field_name)) == 0),
                    lit(None)
                ).otherwise(
                    col(field_name)
                ).alias(field_name)
            return col(field_name).alias(field_name)
            
        elif isinstance(field_type, MapType):
            if field.nullable:
                return when(
                    col(field_name).isNull() | (size(map_keys(col(field_name))) == 0),
                    lit(None)
                ).otherwise(
                    col(field_name)
                ).alias(field_name)
            return col(field_name).alias(field_name)
            
        else:
            return col(field_name).alias(field_name)
    
    existing_columns = {f.name: f for f in df.schema.fields}
    select_expr = []
    
    for field in schema.fields:
        existing_field = existing_columns.get(field.name)
        select_expr.append(build_null_condition(field, "", existing_field))
    
    return df.select(*select_expr)
