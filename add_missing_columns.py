from typing import Union, List
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType, StructField, ArrayType, MapType, StringType, IntegerType, LongType, FloatType,
    DoubleType, BooleanType, BinaryType, TimestampType, DateType, DecimalType, DataType
)
from pyspark.sql.functions import lit, struct, col, transform, expr, when, array, size, map_values

def normalize_schema(schema: DataType) -> DataType:
    """
    Normalize a schema to ensure consistent nullability and remove metadata.
    
    Args:
        schema: PySpark DataType (e.g., StructType, ArrayType, MapType)
        
    Returns:
        DataType: Normalized schema
    """
    if isinstance(schema, StructType):
        return StructType([
            StructField(
                field.name,
                normalize_schema(field.dataType),
                nullable=field.nullable,
                metadata={}
            ) for field in schema.fields
        ])
    elif isinstance(schema, ArrayType):
        return ArrayType(normalize_schema(schema.elementType), containsNull=True)
    elif isinstance(schema, MapType):
        return MapType(normalize_schema(schema.keyType), normalize_schema(schema.valueType), valueContainsNull=True)
    else:
        return schema

def schemas_are_equal(schema1: DataType, schema2: DataType) -> bool:
    """
    Compare two schemas for equality, ignoring metadata.
    
    Args:
        schema1: First PySpark DataType
        schema2: Second PySpark DataType
        
    Returns:
        bool: True if schemas are equal, False otherwise
    """
    return normalize_schema(schema1) == normalize_schema(schema2)

def add_missing_columns(df: DataFrame, target_schema: StructType) -> DataFrame:
    """
    Compare the DataFrame schema with the target schema and add missing columns, including nested fields,
    using df.withColumn. Missing complex columns are set to NULL. For existing structs, maps, and arrays,
    replace empty or all-null values with NULL and adjust field/element/value nullability.
    
    Args:
        df: Input PySpark DataFrame
        target_schema: Target PySpark schema (StructType)
        
    Returns:
        DataFrame: DataFrame with missing columns and nested fields added as NULL
    """
    def _create_null_column(field: StructField):
        """Create a NULL column for a given field, including complex types."""
        return lit(None).cast(field.dataType)

    def _update_struct_column(df: DataFrame, field_name: str, current_field: StructField, 
                             target_field: StructField) -> DataFrame:
        """Update an existing struct column to include missing nested fields and adjust nullability."""
        current_nested_fields = {f.name: f for f in current_field.dataType.fields}
        target_nested_fields = target_field.dataType.fields
        nested_columns = []
        
        for nested_field in target_nested_fields:
            nested_field_name = nested_field.name
            if nested_field_name in current_nested_fields:
                nested_columns.append(
                    col(f"{field_name}.{nested_field_name}").cast(nested_field.dataType).alias(nested_field_name)
                )
            else:
                nested_columns.append(lit(None).cast(nested_field.dataType).alias(nested_field_name))
        
        # Create new struct expression
        new_struct = struct(*nested_columns)
        
        # Replace {NULL, NULL, ...} with NULL
        all_null_condition = " AND ".join(
            f"{field_name}.{nested_field.name} IS NULL" for nested_field in target_nested_fields
        )
        transformed_struct = when(
            expr(all_null_condition),
            lit(None).cast(target_field.dataType)
        ).otherwise(new_struct)
        
        return df.withColumn(field_name, transformed_struct)

    def _update_array_column(df: DataFrame, field_name: str, current_field: StructField, 
                            target_field: StructField) -> DataFrame:
        """Update an existing array column to include missing nested fields and adjust nullability."""
        if schemas_are_equal(current_field.dataType, target_field.dataType):
            return df
        
        if isinstance(target_field.dataType.elementType, StructType) and isinstance(current_field.dataType.elementType, StructType):
            current_nested_fields = {f.name: f for f in current_field.dataType.elementType.fields}
            target_nested_fields = target_field.dataType.elementType.fields
            nested_columns = []
            for nested_field in target_nested_fields:
                nested_field_name = nested_field.name
                if nested_field_name in current_nested_fields:
                    nested_columns.append(f"x.{nested_field_name}")
                else:
                    nested_columns.append(f"CAST(NULL AS {nested_field.dataType.simpleString()}) AS {nested_field_name}")
            
            # Transform array elements
            transform_expr = f"""
                TRANSFORM({field_name}, x -> STRUCT({', '.join(nested_columns)}))
            """
            new_array = expr(transform_expr)
        else:
            # Simple array, cast elements to target type
            new_array = col(field_name)
        
        # Replace [] or all-null arrays with NULL
        all_null_condition = f"""
            {field_name} IS NOT NULL AND
            (SIZE({field_name}) = 0 OR
             SIZE(FILTER({field_name}, x -> x IS NOT NULL)) = 0)
        """
        transformed_array = when(
            expr(all_null_condition),
            lit(None).cast(target_field.dataType)
        ).otherwise(new_array.cast(target_field.dataType))
        
        return df.withColumn(field_name, transformed_array)

    def _update_map_column(df: DataFrame, field_name: str, current_field: StructField, 
                          target_field: StructField) -> DataFrame:
        """Update an existing map column to include missing nested fields or align schema."""
        if schemas_are_equal(current_field.dataType, target_field.dataType):
            return df
        
        if isinstance(target_field.dataType.valueType, StructType) and isinstance(current_field.dataType.valueType, StructType):
            current_nested_fields = {f.name: f for f in current_field.dataType.valueType.fields}
            target_nested_fields = target_field.dataType.valueType.fields
            nested_columns = []
            for nested_field in target_nested_fields:
                nested_field_name = nested_field.name
                if nested_field_name in current_nested_fields:
                    nested_columns.append(f"x.value.{nested_field_name}")
                else:
                    nested_columns.append(f"CAST(NULL AS {nested_field.dataType.simpleString()}) AS {nested_field_name}")
            
            # Transform map values
            transform_expr = f"""
                MAP_FROM_ENTRIES(
                    TRANSFORM(MAP_ENTRIES({field_name}), x -> 
                        STRUCT(x.key AS key, STRUCT({', '.join(nested_columns)}) AS value)
                    )
                )
            """
            new_map = expr(transform_expr)
        else:
            # Simple map, reselect and cast values
            new_map = col(field_name)
        
        # Replace {} or maps with all null values with NULL
        all_null_condition = f"""
            {field_name} IS NOT NULL AND
            (SIZE({field_name}) = 0 OR
             SIZE(FILTER(MAP_VALUES({field_name}), x -> x IS NOT NULL)) = 0)
        """
        transformed_map = when(
            expr(all_null_condition),
            lit(None).cast(target_field.dataType)
        ).otherwise(new_map.cast(target_field.dataType))
        
        return df.withColumn(field_name, transformed_map)

    # Get current DataFrame schema as a dictionary for lookup
    current_fields = {field.name: field for field in df.schema.fields}
    
    # Iterate through target schema fields
    result_df = df
    for target_field in target_schema.fields:
        field_name = target_field.name
        if field_name not in current_fields:
            # Add missing top-level column as NULL
            result_df = result_df.withColumn(field_name, _create_null_column(target_field))
        else:
            # Check for nested field updates
            current_field = current_fields[field_name]
            if isinstance(target_field.dataType, StructType) and isinstance(current_field.dataType, StructType):
                result_df = _update_struct_column(result_df, field_name, current_field, target_field)
            elif isinstance(target_field.dataType, ArrayType) and isinstance(current_field.dataType, ArrayType):
                result_df = _update_array_column(result_df, field_name, current_field, target_field)
            elif isinstance(target_field.dataType, MapType) and isinstance(current_field.dataType, MapType):
                result_df = _update_map_column(result_df, field_name, current_field, target_field)
            # Non-nested fields or type mismatches are preserved as-is
    
    return result_df
