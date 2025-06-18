from typing import Any, Dict, List, Optional
from pyspark.sql import DataFrame, Row
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, DataType
from pyspark.sql.functions import col, lit, when, size, map_keys, array, map_from_arrays, coalesce
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_missing_fields_rdd(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Adds missing fields, including nested fields, to a PySpark DataFrame's RDD based on the schema.
    Sets NULL for all missing fields and sets the uppermost level to NULL if all underlying fields are empty.

    Args:
        df: Input PySpark DataFrame
        schema: Target schema (StructType)
    
    Returns:
        DataFrame with all fields from the schema
    """
    def is_empty(value: Any, data_type: DataType) -> bool:
        """Check if a value is empty (NULL, empty array, empty map, or struct with all empty fields)."""
        if value is None:
            return True
        if isinstance(data_type, StructType):
            if not isinstance(value, (dict, Row)):
                return True
            value_dict = value.asDict() if isinstance(value, Row) else value
            return all(is_empty(value_dict.get(field.name), field.dataType) for field in data_type.fields)
        elif isinstance(data_type, ArrayType):
            return isinstance(value, list) and len(value) == 0
        elif isinstance(data_type, MapType):
            return isinstance(value, dict) and len(value) == 0
        else:
            return value is None

    def align_row(row: Any, schema: StructType) -> Row:
        """Recursively align a Row or dictionary with the schema."""
        if not row:
            return Row(**{field.name: None for field in schema.fields})
        
        # Convert row to dictionary
        data = row.asDict() if isinstance(row, Row) else row if isinstance(row, dict) else {}
        
        new_data = {}
        for field in schema.fields:
            field_name = field.name
            field_type = field.dataType
            value = data.get(field_name)
            
            if value is None:
                new_data[field_name] = None
            elif isinstance(field_type, StructType):
                # Recursively align nested struct
                aligned_struct = align_struct(value, field_type)
                # Set to None if all fields are empty
                new_data[field_name] = None if is_empty(aligned_struct, field_type) else aligned_struct
            elif isinstance(field_type, ArrayType):
                # Process array elements
                if not isinstance(value, list):
                    new_data[field_name] = None
                else:
                    element_type = field_type.elementType
                    if isinstance(element_type, StructType):
                        aligned_elements = [align_struct(elem, element_type) for elem in value]
                        # Set to None if empty
                        new_data[field_name] = None if not aligned_elements else aligned_elements
                    else:
                        new_data[field_name] = None if not value else value
            elif isinstance(field_type, MapType):
                # Process map values
                if not isinstance(value, dict):
                    new_data[field_name] = None
                else:
                    value_type = field_type.valueType
                    if isinstance(value_type, StructType):
                        aligned_values = {k: align_struct(v, value_type) for k, v in value.items()}
                        # Set to None if empty
                        new_data[field_name] = None if not aligned_values else aligned_values
                    else:
                        new_data[field_name] = None if not value else value
            else:
                new_data[field_name] = value
        
        # Set entire struct to None if all fields are empty
        if all(is_empty(new_data[field.name], field.dataType) for field in schema.fields):
            return Row(**{field.name: None for field in schema.fields})
        
        return Row(**new_data)

    def align_struct(value: Any, struct_type: StructType) -> Dict:
        """Align a struct value with the struct schema."""
        if not isinstance(value, (dict, Row)):
            return {field.name: None for field in struct_type.fields}
        
        data = value.asDict() if isinstance(value, Row) else value
        new_data = {}
        
        for field in struct_type.fields:
            field_name = field.name
            field_type = field.dataType
            field_value = data.get(field_name)
            
            if field_value is None:
                new_data[field_name] = None
            elif isinstance(field_type, StructType):
                aligned_struct = align_struct(field_value, field_type)
                new_data[field_name] = None if is_empty(aligned_struct, field_type) else aligned_struct
            elif isinstance(field_type, ArrayType):
                if not isinstance(field_value, list):
                    new_data[field_name] = None
                else:
                    element_type = field_type.elementType
                    if isinstance(element_type, StructType):
                        aligned_elements = [align_struct(elem, element_type) for elem in field_value]
                        new_data[field_name] = None if not aligned_elements else aligned_elements
                    else:
                        new_data[field_name] = None if not field_value else field_value
            elif isinstance(field_type, MapType):
                if not isinstance(field_value, dict):
                    new_data[field_name] = None
                else:
                    value_type = field_type.valueType
                    if isinstance(value_type, StructType):
                        aligned_values = {k: align_struct(v, value_type) for k, v in field_value.items()}
                        new_data[field_name] = None if not aligned_values else aligned_values
                    else:
                        new_data[field_name] = None if not field_value else field_value
            else:
                new_data[field_name] = field_value
        
        return new_data

    # Log schemas
    logger.info(f"Input DataFrame schema: {[f.name for f in df.schema.fields]}")
    logger.info(f"Target schema: {[f.name for f in schema.fields]}")
    
    # Transform RDD
    rdd_aligned = df.rdd.map(lambda row: align_row(row, schema))
    
    # Create new DataFrame
    df_aligned = df.sparkSession.createDataFrame(rdd_aligned, schema)
    
    logger.info(f"Aligned DataFrame schema: {[f.name for f in df_aligned.schema.fields]}")
    
    return df_aligned

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
    logger.info(f"normalize_nulls input schema: {[f.name for f in df.schema.fields]}")
    
    select_expr = []
    
    for field in schema.fields:
        field_name = field.name
        field_type = field.dataType
        
        if isinstance(field_type, (ArrayType, MapType)):
            if isinstance(field_type, ArrayType):
                expr = when(col(field_name).isNull() | (size(col(field_name)) == 0), lit(None)).otherwise(col(field_name))
            else:  # MapType
                expr = when(col(field_name).isNull() | (size(map_keys(col(field_name))) == 0), lit(None)).otherwise(col(field_name))
            select_expr.append(expr.alias(field_name))
        else:
            select_expr.append(col(field_name).alias(field_name))
    
    df_normalized = df.select(*select_expr)
    
    logger.info(f"Normalized DataFrame schema: {[f.name for f in df_normalized.schema.fields]}")
    
    return df_normalized
