from typing import Optional
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, DataType
from pyspark.sql.functions import lit, col, struct, array, map_from_arrays, when, size, map_keys, coalesce

def add_missing_columns(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Adds missing top-level columns to a PySpark DataFrame based on the provided schema.
    Nested structures are initialized with NULL or empty collections, relying on createDataFrame for schema enforcement.
    
    Args:
        df: Input PySpark DataFrame
        schema: Target schema (StructType)
    
    Returns:
        DataFrame with all top-level columns from the schema
    """
    existing_fields = {f.name: f for f in df.schema.fields}
    select_expr = []
    
    for field in schema.fields:
        field_name = field.name
        field_type = field.dataType
        
        if field_name in existing_fields:
            select_expr.append(col(field_name).alias(field_name))
        else:
            if isinstance(field_type, StructType):
                select_expr.append(lit(None).cast(field_type).alias(field_name))
            elif isinstance(field_type, ArrayType):
                select_expr.append(array().cast(field_type).alias(field_name))
            elif isinstance(field_type, MapType):
                select_expr.append(map_from_arrays(array(), array()).cast(field_type).alias(field_name))
            else:
                select_expr.append(lit(None).cast(field_type).alias(field_name))
    
    return df.select(*select_expr)

def normalize_nulls(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Normalizes nullability in a PySpark DataFrame based on the schema.
    Sets empty collections to NULL for nullable fields.
    
    Args:
        df: Input PySpark DataFrame
        schema: Schema (StructType)
    
    Returns:
        DataFrame with normalized NULL values
    """
    select_expr = []
    
    for field in schema.fields:
        field_name = field.name
        field_type = field.dataType
        
        if field.nullable and isinstance(field_type, (ArrayType, MapType)):
            if isinstance(field_type, ArrayType):
                expr = when(col(field_name).isNull() | (size(col(field_name)) == 0), lit(None)).otherwise(col(field_name))
            else:  # MapType
                expr = when(col(field_name).isNull() | (size(map_keys(col(field_name))) == 0), lit(None)).otherwise(col(field_name))
            select_expr.append(expr.alias(field_name))
        else:
            select_expr.append(col(field_name).alias(field_name))
    
    return df.select(*select_expr)
