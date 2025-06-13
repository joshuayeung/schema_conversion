from typing import Union
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType, StructField, ArrayType, MapType, StringType, IntegerType, LongType, FloatType,
    DoubleType, BooleanType, BinaryType, TimestampType, DateType, DecimalType
)
from pyspark.sql.functions import lit, struct, array, create_map

def add_missing_columns(df: DataFrame, target_schema: StructType) -> DataFrame:
    """
    Compare the DataFrame schema with the target schema and add missing columns, including nested structures.
    
    Args:
        df: Input PySpark DataFrame
        target_schema: Target PySpark schema (StructType)
        
    Returns:
        DataFrame: DataFrame with missing columns added as nulls
    """
    def _create_null_column(field: StructField):
        """Create a null column for a given field, handling nested types."""
        if isinstance(field.dataType, StructType):
            # For nested structs, create a struct with null fields
            nested_fields = [
                lit(None).cast(nested_field.dataType).alias(nested_field.name)
                for nested_field in field.dataType.fields
            ]
            return struct(*nested_fields).alias(field.name)
        elif isinstance(field.dataType, ArrayType):
            # For arrays, create an empty array or null array
            return array().cast(field.dataType).alias(field.name)
        elif isinstance(field.dataType, MapType):
            # For maps, create an empty map or null map
            return create_map().cast(field.dataType).alias(field.name)
        else:
            # For primitive types, use lit(None) with the correct type
            return lit(None).cast(field.dataType).alias(field.name)

    def _compare_and_add_fields(current_fields: dict, target_fields: list, parent_path: str = "") -> list:
        """
        Compare current and target fields, adding missing ones.
        
        Args:
            current_fields: Dict of current field names to StructField (for lookup)
            target_fields: List of target StructField objects
            parent_path: Path to the current field (for nested fields)
        
        Returns:
            List of columns to select (existing or new null columns)
        """
        select_columns = []
        
        for target_field in target_fields:
            field_name = target_field.name
            full_path = f"{parent_path}.{field_name}" if parent_path else field_name
            
            if field_name not in current_fields:
                # Field is missing, add it as a null column
                select_columns.append(_create_null_column(target_field))
            else:
                # Field exists, check if it needs nested updates
                current_field = current_fields[field_name]
                if isinstance(target_field.dataType, StructType) and isinstance(current_field.dataType, StructType):
                    # Handle nested structs
                    nested_current_fields = {f.name: f for f in current_field.dataType.fields}
                    nested_target_fields = target_field.dataType.fields
                    nested_columns = _compare_and_add_fields(nested_current_fields, nested_target_fields, full_path)
                    select_columns.append(struct(*nested_columns).alias(field_name))
                elif isinstance(target_field.dataType, ArrayType) and isinstance(current_field.dataType, ArrayType):
                    # Handle arrays with nested structs
                    if isinstance(target_field.dataType.elementType, StructType) and isinstance(current_field.dataType.elementType, StructType):
                        nested_current_fields = {f.name: f for f in current_field.dataType.elementType.fields}
                        nested_target_fields = target_field.dataType.elementType.fields
                        nested_columns = _compare_and_add_fields(nested_current_fields, nested_target_fields, f"{full_path}[]")
                        select_columns.append(
                            array(struct(*nested_columns)).cast(target_field.dataType).alias(field_name)
                        )
                    else:
                        # Non-struct array, keep as is
                        select_columns.append(field_name)
                elif isinstance(target_field.dataType, MapType) and isinstance(current_field.dataType, MapType):
                    # Handle maps with nested structs as values
                    if isinstance(target_field.dataType.valueType, StructType) and isinstance(current_field.dataType.valueType, StructType):
                        nested_current_fields = {f.name: f for f in current_field.dataType.valueType.fields}
                        nested_target_fields = target_field.dataType.valueType.fields
                        nested_columns = _compare_and_add_fields(nested_current_fields, nested_target_fields, f"{full_path}[]")
                        select_columns.append(
                            create_map(lit("key"), struct(*nested_columns)).cast(target_field.dataType).alias(field_name)
                        )
                    else:
                        # Non-struct map, keep as is
                        select_columns.append(field_name)
                else:
                    # Non-nested field or type mismatch, keep as is
                    select_columns.append(field_name)
        
        return select_columns

    # Get current DataFrame schema as a dictionary for lookup
    current_fields = {field.name: field for field in df.schema.fields}
    
    # Compare and add missing fields
    select_columns = _compare_and_add_fields(current_fields, target_schema.fields)
    
    # Select columns to create the new DataFrame
    return df.select(*select_columns)
