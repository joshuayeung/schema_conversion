from typing import Union, List
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType, StructField, ArrayType, MapType, StringType, IntegerType, LongType, FloatType,
    DoubleType, BooleanType, BinaryType, TimestampType, DateType, DecimalType, DataType
)
from pyspark.sql.functions import lit, struct, col, transform, expr

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
                nullable=True,
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
    Compare two schemas for equality, ignoring metadata and normalizing nullability.
    
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
    using df.withColumn. Missing complex columns are set to NULL.
    
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
        """Update an existing struct column to include missing nested fields, handling nested complex types."""
        current_nested_fields = {f.name: f for f in current_field.dataType.fields}
        target_nested_fields = target_field.dataType.fields
        nested_columns = []
        
        for nested_field in target_nested_fields:
            nested_field_name = nested_field.name
            if nested_field_name in current_nested_fields:
                current_nested_field = current_nested_fields[nested_field_name]
                if isinstance(nested_field.dataType, StructType) and isinstance(current_nested_field.dataType, StructType):
                    # Recursively update nested struct
                    temp_col_name = f"__temp_{field_name}_{nested_field_name}"
                    df = _update_struct_column(df, f"{field_name}.{nested_field_name}", current_nested_field, nested_field)
                    nested_columns.append(col(f"{field_name}.{nested_field_name}").alias(nested_field_name))
                elif isinstance(nested_field.dataType, ArrayType) and isinstance(current_nested_field.dataType, ArrayType):
                    # Update nested array
                    df = _update_array_column(df, f"{field_name}.{nested_field_name}", current_nested_field, nested_field)
                    nested_columns.append(col(f"{field_name}.{nested_field_name}").alias(nested_field_name))
                elif isinstance(nested_field.dataType, MapType) and isinstance(current_nested_field.dataType, MapType):
                    # Update nested map
                    df = _update_map_column(df, f"{field_name}.{nested_field_name}", current_nested_field, nested_field)
                    nested_columns.append(col(f"{field_name}.{nested_field_name}").alias(nested_field_name))
                else:
                    # Preserve existing primitive or non-matching field
                    nested_columns.append(col(f"{field_name}.{nested_field_name}").alias(nested_field_name))
            else:
                # Add missing nested field as NULL
                nested_columns.append(lit(None).cast(nested_field.dataType).alias(nested_field_name))
        
        # Create new struct column with all fields
        return df.withColumn(field_name, struct(*nested_columns))

    def _update_array_column(df: DataFrame, field_name: str, current_field: StructField, 
                            target_field: StructField) -> DataFrame:
        """Update an existing array column to include missing nested fields in its elements, handling nested complex types."""
        if schemas_are_equal(current_field.dataType, target_field.dataType):
            return df
        
        if isinstance(target_field.dataType.elementType, StructType) and isinstance(current_field.dataType.elementType, StructType):
            current_nested_fields = {f.name: f for f in current_field.dataType.elementType.fields}
            target_nested_fields = target_field.dataType.elementType.fields
            nested_columns = []
            
            for nested_field in target_nested_fields:
                nested_field_name = nested_field.name
                if nested_field_name in current_nested_fields:
                    current_nested_field = current_nested_fields[nested_field_name]
                    if isinstance(nested_field.dataType, StructType) and isinstance(current_nested_field.dataType, StructType):
                        # Create a temporary array with updated structs
                        temp_array_col = f"__temp_array_{field_name}_{nested_field_name}"
                        df = _update_struct_column(df, f"{field_name}[*].{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.{nested_field_name}")
                    elif isinstance(nested_field.dataType, ArrayType) and isinstance(current_nested_field.dataType, ArrayType):
                        # Create a temporary array with updated arrays
                        temp_array_col = f"__temp_array_{field_name}_{nested_field_name}"
                        df = _update_array_column(df, f"{field_name}[*].{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.{nested_field_name}")
                    elif isinstance(nested_field.dataType, MapType) and isinstance(current_nested_field.dataType, MapType):
                        # Create a temporary array with updated maps
                        temp_array_col = f"__temp_array_{field_name}_{nested_field_name}"
                        df = _update_map_column(df, f"{field_name}[*].{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.{nested_field_name}")
                    else:
                        nested_columns.append(f"x.{nested_field_name}")
                else:
                    nested_columns.append(f"CAST(NULL AS {nested_field.dataType.simpleString()}) AS {nested_field_name}")
            
            # Transform array elements using expr
            transform_expr = f"""
                TRANSFORM({field_name}, x -> STRUCT({', '.join(nested_columns)}))
            """
            return df.withColumn(field_name, expr(transform_expr).cast(target_field.dataType))
        elif isinstance(target_field.dataType.elementType, ArrayType) and isinstance(current_field.dataType.elementType, ArrayType):
            # Handle nested arrays
            temp_array_col = f"__temp_array_{field_name}"
            df = df.withColumn(temp_array_col, transform(col(field_name), lambda x: col(x)))
            df = _update_array_column(df, temp_array_col, StructField(temp_array_col, current_field.dataType.elementType), 
                                     StructField(temp_array_col, target_field.dataType.elementType))
            return df.withColumn(field_name, col(temp_array_col).cast(target_field.dataType)).drop(temp_array_col)
        elif isinstance(target_field.dataType.elementType, MapType) and isinstance(current_field.dataType.elementType, MapType):
            # Handle nested maps
            temp_array_col = f"__temp_array_{field_name}"
            df = df.withColumn(temp_array_col, transform(col(field_name), lambda x: col(x)))
            df = _update_map_column(df, temp_array_col, StructField(temp_array_col, current_field.dataType.elementType), 
                                   StructField(temp_array_col, target_field.dataType.elementType))
            return df.withColumn(field_name, col(temp_array_col).cast(target_field.dataType)).drop(temp_array_col)
        else:
            # Non-complex array, cast to target type
            return df.withColumn(field_name, col(field_name).cast(target_field.dataType))

    def _update_map_column(df: DataFrame, field_name: str, current_field: StructField, 
                          target_field: StructField) -> DataFrame:
        """Update an existing map column to include missing nested fields in its values, handling nested complex types."""
        if schemas_are_equal(current_field.dataType, target_field.dataType):
            return df
        
        if isinstance(target_field.dataType.valueType, StructType) and isinstance(current_field.dataType.valueType, StructType):
            current_nested_fields = {f.name: f for f in current_field.dataType.valueType.fields}
            target_nested_fields = target_field.dataType.valueType.fields
            nested_columns = []
            
            for nested_field in target_nested_fields:
                nested_field_name = nested_field.name
                if nested_field_name in current_nested_fields:
                    current_nested_field = current_nested_fields[nested_field_name]
                    if isinstance(nested_field.dataType, StructType) and isinstance(current_nested_field.dataType, StructType):
                        # Create a temporary map with updated structs
                        temp_map_col = f"__temp_map_{field_name}_{nested_field_name}"
                        df = _update_struct_column(df, f"{field_name}.value.{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.value.{nested_field_name}")
                    elif isinstance(nested_field.dataType, ArrayType) and isinstance(current_nested_field.dataType, ArrayType):
                        # Create a temporary map with updated arrays
                        temp_map_col = f"__temp_map_{field_name}_{nested_field_name}"
                        df = _update_array_column(df, f"{field_name}.value.{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.value.{nested_field_name}")
                    elif isinstance(nested_field.dataType, MapType) and isinstance(current_nested_field.dataType, MapType):
                        # Create a temporary map with updated maps
                        temp_map_col = f"__temp_map_{field_name}_{nested_field_name}"
                        df = _update_map_column(df, f"{field_name}.value.{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.value.{nested_field_name}")
                    else:
                        nested_columns.append(f"x.value.{nested_field_name}")
                else:
                    nested_columns.append(f"CAST(NULL AS {nested_field.dataType.simpleString()}) AS {nested_field_name}")
            
            # Transform map values using expr
            transform_expr = f"""
                MAP_FROM_ENTRIES(
                    TRANSFORM(MAP_ENTRIES({field_name}), x -> 
                        STRUCT(x.key AS key, STRUCT({', '.join(nested_columns)}) AS value)
                    )
                )
            """
            return df.withColumn(field_name, expr(transform_expr).cast(target_field.dataType))
        elif isinstance(target_field.dataType.valueType, ArrayType) and isinstance(current_field.dataType.valueType, ArrayType):
            # Handle nested arrays as map values
            temp_map_col = f"__temp_map_{field_name}"
            df = df.withColumn(temp_map_col, expr(f"""
                MAP_FROM_ENTRIES(
                    TRANSFORM(MAP_ENTRIES({field_name}), x -> STRUCT(x.key AS key, x.value AS value))
                )
            """))
            df = _update_array_column(df, f"{temp_map_col}.value", StructField("value", current_field.dataType.valueType), 
                                     StructField("value", target_field.dataType.valueType))
            return df.withColumn(field_name, col(temp_map_col).cast(target_field.dataType)).drop(temp_map_col)
        elif isinstance(target_field.dataType.valueType, MapType) and isinstance(current_field.dataType.valueType, MapType):
            # Handle nested maps as map values
            temp_map_col = f"__temp_map_{field_name}"
            df = df.withColumn(temp_map_col, expr(f"""
                MAP_FROM_ENTRIES(
                    TRANSFORM(MAP_ENTRIES({field_name}), x -> STRUCT(x.key AS key, x.value AS value))
                )
            """))
            df = _update_map_column(df, f"{temp_map_col}.value", StructField("value", current_field.dataType.valueType), 
                                   StructField("value", target_field.dataType.valueType))
            return df.withColumn(field_name, col(temp_map_col).cast(target_field.dataType)).drop(temp_map_col)
        else:
            # Simple map, reselect to avoid cast issues
            return df.withColumn(field_name, col(field_name))

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
