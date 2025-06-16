from typing import Union, List
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType, StructField, ArrayType, MapType, StringType, IntegerType, LongType, FloatType,
    DoubleType, BooleanType, BinaryType, TimestampType, DateType, DecimalType, DataType
)
from pyspark.sql.functions import lit, struct, col, transform, expr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    using df.withColumn. Missing complex columns are set to NULL or constructed from flattened fields.
    
    Args:
        df: Input PySpark DataFrame
        target_schema: Target PySpark schema (StructType)
        
    Returns:
        DataFrame: DataFrame with missing columns and nested fields added correctly
    """
    def _create_null_column(field: StructField):
        """Create a NULL column for a given field, ensuring StructType is not flattened."""
        logger.info(f"Creating NULL column for field: {field.name} with type {field.dataType.simpleString()}")
        if isinstance(field.dataType, StructType):
            return lit(None).cast(field.dataType)
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
                    temp_col_name = f"__temp_{field_name}_{nested_field_name}"
                    df = _update_struct_column(df, f"{field_name}.{nested_field_name}", current_nested_field, nested_field)
                    nested_columns.append(col(f"{field_name}.{nested_field_name}").alias(nested_field_name))
                elif isinstance(nested_field.dataType, ArrayType) and isinstance(current_nested_field.dataType, ArrayType):
                    df = _update_array_column(df, f"{field_name}.{nested_field_name}", current_nested_field, nested_field)
                    nested_columns.append(col(f"{field_name}.{nested_field_name}").alias(nested_field_name))
                elif isinstance(nested_field.dataType, MapType) and isinstance(current_nested_field.dataType, MapType):
                    df = _update_map_column(df, f"{field_name}.{nested_field_name}", current_nested_field, nested_field)
                    nested_columns.append(col(f"{field_name}.{nested_field_name}").alias(nested_field_name))
                else:
                    nested_columns.append(col(f"{field_name}.{nested_field_name}").alias(nested_field_name))
            else:
                nested_columns.append(lit(None).cast(nested_field.dataType).alias(nested_field_name))
        
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
                        temp_array_col = f"__temp_array_{field_name}_{nested_field_name}"
                        df = _update_struct_column(df, f"{field_name}[*].{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.{nested_field_name}")
                    elif isinstance(nested_field.dataType, ArrayType) and isinstance(current_nested_field.dataType, ArrayType):
                        temp_array_col = f"__temp_array_{field_name}_{nested_field_name}"
                        df = _update_array_column(df, f"{field_name}[*].{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.{nested_field_name}")
                    elif isinstance(nested_field.dataType, MapType) and isinstance(current_nested_field.dataType, MapType):
                        temp_array_col = f"__temp_array_{field_name}_{nested_field_name}"
                        df = _update_map_column(df, f"{field_name}[*].{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.{nested_field_name}")
                    else:
                        nested_columns.append(f"x.{nested_field_name}")
                else:
                    nested_columns.append(f"CAST(NULL AS {nested_field.dataType.simpleString()}) AS {nested_field_name}")
            
            transform_expr = f"""
                TRANSFORM({field_name}, x -> STRUCT({', '.join(nested_columns)}))
            """
            return df.withColumn(field_name, expr(transform_expr).cast(target_field.dataType))
        elif isinstance(target_field.dataType.elementType, ArrayType) and isinstance(current_field.dataType.elementType, ArrayType):
            temp_array_col = f"__temp_array_{field_name}"
            df = df.withColumn(temp_array_col, transform(col(field_name), lambda x: col(x)))
            df = _update_array_column(df, temp_array_col, StructField(temp_array_col, current_field.dataType.elementType), 
                                     StructField(temp_array_col, target_field.dataType.elementType))
            return df.withColumn(field_name, col(temp_array_col).cast(target_field.dataType)).drop(temp_array_col)
        elif isinstance(target_field.dataType.elementType, MapType) and isinstance(current_field.dataType.elementType, MapType):
            temp_array_col = f"__temp_array_{field_name}"
            df = df.withColumn(temp_array_col, transform(col(field_name), lambda x: col(x)))
            df = _update_map_column(df, temp_array_col, StructField(temp_array_col, current_field.dataType.elementType), 
                                   StructField(temp_array_col, target_field.dataType.elementType))
            return df.withColumn(field_name, col(temp_array_col).cast(target_field.dataType)).drop(temp_array_col)
        else:
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
                        temp_map_col = f"__temp_map_{field_name}_{nested_field_name}"
                        df = _update_struct_column(df, f"{field_name}.value.{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.value.{nested_field_name}")
                    elif isinstance(nested_field.dataType, ArrayType) and isinstance(current_nested_field.dataType, ArrayType):
                        temp_map_col = f"__temp_map_{field_name}_{nested_field_name}"
                        df = _update_array_column(df, f"{field_name}.value.{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.value.{nested_field_name}")
                    elif isinstance(nested_field.dataType, MapType) and isinstance(current_nested_field.dataType, MapType):
                        temp_map_col = f"__temp_map_{field_name}_{nested_field_name}"
                        df = _update_map_column(df, f"{field_name}.value.{nested_field_name}", current_nested_field, nested_field)
                        nested_columns.append(f"x.value.{nested_field_name}")
                    else:
                        nested_columns.append(f"x.value.{nested_field_name}")
                else:
                    nested_columns.append(f"CAST(NULL AS {nested_field.dataType.simpleString()}) AS {nested_field_name}")
            
            transform_expr = f"""
                MAP_FROM_ENTRIES(
                    TRANSFORM(MAP_ENTRIES({field_name}), x -> 
                        STRUCT(x.key AS key, STRUCT({', '.join(nested_columns)}) AS value)
                    )
                )
            """
            return df.withColumn(field_name, expr(transform_expr).cast(target_field.dataType))
        elif isinstance(target_field.dataType.valueType, ArrayType) and isinstance(current_field.dataType.valueType, ArrayType):
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
            return df.withColumn(field_name, col(field_name))

    # Log target schema for debugging
    logger.info("Target schema fields:")
    for field in target_schema.fields:
        logger.info(f"Field: {field.name}, Type: {field.dataType.simpleString()}")

    # Validate target schema to prevent flattened fields
    for field in target_schema.fields:
        if "." in field.name:
            logger.warning(f"Field name contains dot: {field.name}. This may indicate schema flattening.")

    # Log input DataFrame schema
    logger.info(f"Input DataFrame schema: {df.schema.simpleString()}")

    # Detect and consolidate flattened columns for StructType fields
    current_columns = [field.name for field in df.schema.fields]
    result_df = df
    columns_to_drop = []
    for target_field in target_schema.fields:
        field_name = target_field.name
        if isinstance(target_field.dataType, StructType):
            # Check for flattened columns (e.g., lastAction.authorization)
            flattened_columns = {}
            for col_name in current_columns:
                if col_name.startswith(f"{field_name}."):
                    nested_field_name = col_name[len(field_name) + 1:]
                    flattened_columns[nested_field_name] = col_name
            
            if flattened_columns:
                logger.info(f"Flattened columns detected for {field_name}: {list(flattened_columns.values())}")
                # Construct the StructType column
                nested_columns = []
                for nested_field in target_field.dataType.fields:
                    nested_field_name = nested_field.name
                    if nested_field_name in flattened_columns:
                        # Use existing flattened column
                        col_name = flattened_columns[nested_field_name]
                        nested_columns.append(col(col_name).cast(nested_field.dataType).alias(nested_field_name))
                        columns_to_drop.append(col_name)
                    else:
                        # Add missing field as NULL
                        nested_columns.append(lit(None).cast(nested_field.dataType).alias(nested_field_name))
                
                # Add the StructType column
                logger.info(f"Creating StructType column {field_name} from flattened columns")
                result_df = result_df.withColumn(field_name, struct(*nested_columns))
                logger.info(f"Schema after adding {field_name}: {result_df.schema.simpleString()}")
                continue  # Skip further processing for this field
        
        # Handle non-StructType fields or StructType fields without flattened columns
        if field_name not in [f.name for f in result_df.schema.fields]:
            logger.info(f"Adding missing column: {field_name}")
            result_df = result_df.withColumn(field_name, _create_null_column(target_field))
            logger.info(f"Schema after adding {field_name}: {result_df.schema.simpleString()}")
        else:
            # Check for nested field updates
            current_field = next(f for f in result_df.schema.fields if f.name == field_name)
            if isinstance(target_field.dataType, StructType) and isinstance(current_field.dataType, StructType):
                result_df = _update_struct_column(result_df, field_name, current_field, target_field)
            elif isinstance(target_field.dataType, ArrayType) and isinstance(current_field.dataType, ArrayType):
                result_df = _update_array_column(result_df, field_name, current_field, target_field)
            elif isinstance(target_field.dataType, MapType) and isinstance(current_field.dataType, MapType):
                result_df = _update_map_column(result_df, field_name, current_field, target_field)
    
    # Drop flattened columns
    if columns_to_drop:
        logger.info(f"Dropping flattened columns: {columns_to_drop}")
        result_df = result_df.drop(*columns_to_drop)
        logger.info(f"Schema after dropping flattened columns: {result_df.schema.simpleString()}")

    # Final schema validation
    final_columns = [field.name for field in result_df.schema.fields]
    for target_field in target_schema.fields:
        if isinstance(target_field.dataType, StructType):
            for nested_field in target_field.dataType.fields:
                conflicting_name = f"{target_field.name}.{nested_field.name}"
                if conflicting_name in final_columns:
                    logger.error(f"Flattened column detected: {conflicting_name}. Expected single column {target_field.name}")
                    raise ValueError(f"Flattened column {conflicting_name} found in output schema")

    logger.info(f"Final DataFrame schema: {result_df.schema.simpleString()}")
    return result_df
