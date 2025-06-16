from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, NullType
from pyspark.sql.functions import lit, col, when, isnull, array, struct, map_from_arrays

def add_missing_columns(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Adds missing columns to a PySpark DataFrame based on the provided schema.
    Missing columns (including nested structs, maps, arrays) are added as NULL.
    
    Args:
        df: Input PySpark DataFrame
        schema: Target schema (StructType) to align the DataFrame with
    
    Returns:
        DataFrame with all columns from the schema, missing ones added as NULL
    """
    def get_null_expr(field: StructField, prefix: str = "") -> str:
        """Generate expression for NULL values based on field type."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix else field.name
        
        if isinstance(field_type, StructType):
            # For structs, create a struct with all fields as NULL
            subfields = [get_null_expr(subfield, f"{field_name}.") for subfield in field_type.fields]
            return struct(*subfields).alias(field_name)
        elif isinstance(field_type, ArrayType):
            # For arrays, create an empty array of the element type
            return array().cast(field_type).alias(field_name)
        elif isinstance(field_type, MapType):
            # For maps, create an empty map
            return map_from_arrays(array(), array()).cast(field_type).alias(field_name)
        else:
            # For primitive types, use NULL
            return lit(None).cast(field_type).alias(field_name)
    
    # Get current DataFrame columns
    existing_columns = set(df.columns)
    select_expr = [col(c) for c in df.columns]
    
    # Add missing columns from schema
    for field in schema.fields:
        if field.name not in existing_columns:
            select_expr.append(get_null_expr(field))
    
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
    def build_null_condition(field: StructField, prefix: str = "") -> str:
        """Build condition to check if a field should be set to NULL."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix else field.name
        
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
                
                # Recursively process subfields
                subfields = [
                    build_null_condition(subfield, f"{field_name}.")
                    for subfield in field_type.fields
                ]
                return when(
                    combined_condition,
                    lit(None)
                ).otherwise(
                    struct(*[col(f"{field_name}.{subfield.name}").alias(subfield.name)
                            for subfield in field_type.fields])
                ).alias(field_name)
            else:
                # If struct is not nullable, just process subfields
                subfields = [
                    build_null_condition(subfield, f"{field_name}.")
                    for subfield in field_type.fields
                ]
                return struct(*subfields).alias(field_name)
                
        elif isinstance(field_type, ArrayType):
            # For arrays, process each element recursively
            if field.nullable and isinstance(field_type.elementType, StructType):
                # Create a temporary column name for the array elements
                element_alias = f"{field_name}_element"
                element_condition = build_null_condition(
                    StructField(element_alias, field_type.elementType, True)
                )
                return when(
                    col(field_name).isNull(),
                    lit(None)
                ).otherwise(
                    array(*[element_condition]).cast(field_type)
                ).alias(field_name)
            return col(field_name)
            
        elif isinstance(field_type, MapType):
            # For maps, process values recursively
            if field.nullable and isinstance(field_type.valueType, StructType):
                # Create a temporary column name for map values
                value_alias = f"{field_name}_value"
                value_condition = build_null_condition(
                    StructField(value_alias, field_type.valueType, True)
                )
                return when(
                    col(field_name).isNull(),
                    lit(None)
                ).otherwise(
                    map_from_arrays(
                        array(lit("key")),
                        array(value_condition)
                    ).cast(field_type)
                ).alias(field_name)
            return col(field_name)
            
        else:
            # For primitive types, return as is
            return col(field_name)
    
    # Build select expressions for all fields
    select_expr = [build_null_condition(field) for field in schema.fields]
    
    return df.select(*select_expr)
