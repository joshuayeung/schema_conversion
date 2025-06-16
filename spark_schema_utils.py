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
    def get_field_expr(field: StructField, prefix: str = "", existing_field: StructField = None) -> str:
        """Generate expression for a field, handling nested structures and existing fields."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix else field.name
        
        if isinstance(field_type, StructType):
            # Check if the struct exists in the DataFrame
            existing_struct_type = existing_field.dataType if existing_field and isinstance(existing_field.dataType, StructType) else None
            existing_subfields = {f.name: f for f in existing_struct_type.fields} if existing_struct_type else {}
            
            # Generate expressions for subfields
            subfield_exprs = []
            for subfield in field_type.fields:
                existing_subfield = existing_subfields.get(subfield.name)
                subfield_expr = get_field_expr(
                    subfield,
                    f"{field_name}." if field_name else "",
                    existing_subfield
                )
                subfield_expr = subfield_expr.alias(subfield.name)
                subfield_exprs.append(subfield_expr)
            
            # If the struct exists, preserve existing subfields and add missing ones
            if existing_field:
                return when(
                    col(field_name).isNotNull(),
                    struct(*subfield_exprs)
                ).otherwise(
                    struct(*subfield_exprs)
                ).alias(field_name)
            else:
                # If the struct is missing entirely, create it with all subfields as NULL
                return struct(*subfield_exprs).alias(field_name)
                
        elif isinstance(field_type, ArrayType):
            if existing_field and isinstance(existing_field.dataType, ArrayType):
                # Preserve existing array if it exists
                return col(field_name)
            # Create an empty array of the correct type
            return array().cast(field_type).alias(field_name)
            
        elif isinstance(field_type, MapType):
            if existing_field and isinstance(field_type, MapType):
                # Preserve existing map if it exists
                return col(field_name)
            # Create an empty map of the correct type
            return map_from_arrays(array(), array()).cast(field_type).alias(field_name)
            
        else:
            # For primitive types, use existing value or NULL
            if existing_field:
                return col(field_name)
            return lit(None).cast(field_type).alias(field_name)
    
    # Get existing columns and their types
    existing_columns = {f.name: f for f in df.schema.fields}
    select_expr = []
    
    # Process each field in the schema
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
    def build_null_condition(field: StructField, prefix: str = "", is_array_element: bool = False) -> str:
        """Build condition to check if a field should be set to NULL."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix and not is_array_element else field.name
        
        if isinstance(field_type, StructType):
            # For structs, check if all required fields are NULL
            required_fields = [f for f in field_type.fields if not f.nullable]
            if not required_fields:
                return col(field_name)
            
            # If the struct is nullable, check if all required fields are NULL
            if field.nullable:
                conditions = [col(f"{field_name}.{f.name}").isNull() for f in required_fields]
                combined_condition = conditions[0]
                for cond in conditions[1:]:
                    combined_condition = combined_condition & cond
                
                # Process subfields without including parent prefix in their names
                subfields = [
                    build_null_condition(subfield, f"{field_name}.", is_array_element=is_array_element).alias(subfield.name)
                    for subfield in field_type.fields
                ]
                return when(
                    combined_condition,
                    lit(None)
                ).otherwise(
                    struct(*subfields)
                ).alias(field_name)
            else:
                # If struct is not nullable, just process subfields
                subfields = [
                    build_null_condition(subfield, f"{field_name}.", is_array_element=is_array_element).alias(subfield.name)
                    for subfield in field_type.fields
                ]
                return struct(*subfields).alias(field_name)
                
        elif isinstance(field_type, ArrayType):
            # For arrays, process each element recursively
            if field.nullable and isinstance(field_type.elementType, StructType):
                # Process each array element using transform
                element_condition = build_null_condition(
                    StructField(field.name, field_type.elementType, True),
                    prefix="",
                    is_array_element=True
                )
                return when(
                    col(field_name).isNull(),
                    lit(None)
                ).otherwise(
                    transform(col(field_name), lambda x: element_condition).cast(field_type)
                ).alias(field_name)
            return col(field_name)
            
        elif isinstance(field_type, MapType):
            # For maps, process values recursively
            if field.nullable and isinstance(field_type.valueType, StructType):
                # Process map values using transform
                value_condition = build_null_condition(
                    StructField(field.name, field_type.valueType, True),
                    prefix="",
                    is_array_element=True
                )
                return when(
                    col(field_name).isNull(),
                    lit(None)
                ).otherwise(
                    expr(f"transform_values({field_name}, (k, v) -> {value_condition})").cast(field_type)
                ).alias(field_name)
            return col(field_name)
            
        else:
            # For primitive types, return as is
            return col(field_name)
    
    # Build select expressions for all fields
    select_expr = [build_null_condition(field) for field in schema.fields]
    
    return df.select(*select_expr)
