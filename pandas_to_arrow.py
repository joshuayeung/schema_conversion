import pandas as pd
import pyarrow as pa
from typing import Any, List, Dict
import numpy as np
from pyspark.sql.types import Row

def pandas_to_arrow_with_nested_schema(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """
    Convert a pandas DataFrame to a PyArrow Table, aligning with a schema that may include
    multi-level nested types (struct, list, map_). Missing columns are filled with nulls.
    Supports PySpark Row objects for struct fields.
    
    Args:
        df (pd.DataFrame): Input pandas DataFrame, possibly containing Row objects.
        schema (pa.Schema): Target PyArrow schema with possible nested types.
    
    Returns:
        pa.Table: PyArrow Table conforming to the provided schema.
    
    Raises:
        ValueError: If DataFrame contains columns not in schema or data is incompatible.
    """
    # Validate DataFrame columns against schema
    df_columns = set(df.columns)
    schema_names = {field.name for field in schema}
    extra_columns = df_columns - schema_names
    if extra_columns:
        raise ValueError(f"DataFrame contains columns not in schema: {extra_columns}")

    # Initialize list to store PyArrow arrays for each field
    arrays = []
    
    for field in schema:
        field_name = field.name
        field_type = field.type
        
        if field_name in df:
            # Column exists in DataFrame, convert to PyArrow array
            column_data = df[field_name]
            array = _convert_to_arrow_array(column_data, field_type, field_name)
        else:
            # Column missing, create null array
            array = pa.array([None] * len(df), type=field_type)
        
        arrays.append(array)
    
    # Create PyArrow Table from arrays
    return pa.Table.from_arrays(arrays, schema=schema)

def _convert_to_arrow_array(series: pd.Series, field_type: pa.DataType, field_name: str) -> pa.Array:
    """
    Convert a pandas Series to a PyArrow array, handling nested types recursively.
    
    Args:
        series: Pandas Series containing the data.
        field_type: PyArrow data type (may be struct, list, map_, or primitive).
        field_name: Name of the field for error reporting.
    
    Returns:
        pa.Array: PyArrow array matching the field type.
    
    Raises:
        ValueError: If data is incompatible with the field type.
    """
    # Replace pandas NA/NaN with None
    series = series.where(~series.isna(), None)
    
    if series.isna().all() or series.isnull().all():
        return pa.array([None] * len(series), type=field_type)
    
    if pa.types.is_struct(field_type):
        return _convert_to_struct_array(series, field_type, field_name)
    elif pa.types.is_list(field_type):
        return _convert_to_list_array(series, field_type, field_name)
    elif pa.types.is_map(field_type):
        return _convert_to_map_array(series, field_type, field_name)
    else:
        # Handle primitive types
        try:
            return pa.array(series, type=field_type)
        except Exception as e:
            raise ValueError(f"Failed to convert column '{field_name}' to {field_type}: {str(e)}")

def _convert_to_struct_array(series: pd.Series, struct_type: pa.StructType, field_name: str) -> pa.Array:
    """
    Convert a pandas Series to a PyArrow struct array, handling nested fields recursively.
    Supports PySpark Row objects and dictionaries.
    """
    # Replace pandas NA/NaN with None
    series = series.where(~series.isna(), None)
    
    if series.isna().all() or series.isnull().all():
        return pa.array([None] * len(series), type=struct_type)
    
    # Extract struct fields
    struct_fields = {f.name: f.type for f in struct_type}
    records = []
    
    for item in series:
        if item is None:
            records.append(None)
        else:
            # Convert PySpark Row to dict if necessary
            if isinstance(item, Row):
                item = item.asDict(recursive=True)
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict or Row for struct field '{field_name}', got {type(item)} in {item}")
            record = {}
            for subfield_name, subfield_type in struct_fields.items():
                value = item.get(subfield_name, None)
                if value is not None and (pa.types.is_struct(subfield_type) or pa.types.is_list(subfield_type) or pa.types.is_map(subfield_type)):
                    # Recursively convert nested types
                    sub_series = pd.Series([value]).where(~pd.Series([value]).isna(), None)
                    sub_array = _convert_to_arrow_array(sub_series, subfield_type, f"{field_name}.{subfield_name}")
                    record[subfield_name] = sub_array[0]
                else:
                    record[subfield_name] = value
            records.append(record)
    
    try:
        return pa.array(records, type=struct_type)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{field_name}' to struct {struct_type}: {str(e)}")

def _convert_to_list_array(series: pd.Series, list_type: pa.ListType, field_name: str) -> pa.Array:
    """
    Convert a pandas Series to a PyArrow list array, handling nested value types recursively.
    Assumes Series contains lists, tuples, or None.
    """
    # Replace pandas NA/NaN with None
    series = series.where(~series.isna(), None)
    
    if series.isna().all() or series.isnull().all():
        return pa.array([None] * len(series), type=list_type)
    
    value_type = list_type.value_type
    data = []
    
    for item in series:
        if item is None:
            data.append(None)
        elif not isinstance(item, (list, tuple)):
            raise ValueError(f"Expected list or tuple for list field '{field_name}', got {type(item)} in {item}")
        else:
            if pa.types.is_struct(value_type) or pa.types.is_list(value_type) or pa.types.is_map(value_type):
                # Recursively convert each element in the list
                if len(item) == 0:
                    data.append([])
                else:
                    sub_series = pd.Series(item).where(~pd.Series(item).isna(), None)
                    sub_array = _convert_to_arrow_array(sub_series, value_type, f"{field_name}.list")
                    data.append(sub_array)
            else:
                data.append(item)
    
    try:
        return pa.array(data, type=list_type)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{field_name}' to list {list_type}: {str(e)}")

def _convert_to_map_array(series: pd.Series, map_type: pa.MapType, field_name: str) -> pa.Array:
    """
    Convert a pandas Series to a PyArrow map array, handling nested key/value types recursively.
    Assumes Series contains dictionaries, lists of tuples, or None.
    """
    # Replace pandas NA/NaN with None
    series = series.where(~series.isna(), None)
    
    if series.isna().all() or series.isnull().all():
        return pa.array([None] * len(series), type=map_type)
    
    key_type = map_type.key_type
    value_type = map_type.item_type
    data = []
    
    for item in series:
        if item is None:
            data.append(None)
        else:
            if isinstance(item, dict):
                item = list(item.items())
            if not isinstance(item, (list, tuple)):
                raise ValueError(f"Expected dict or list of tuples for map field '{field_name}', got {type(item)} in {item}")
            
            # Process key-value pairs
            processed_pairs = []
            for key, value in item:
                if pa.types.is_struct(value_type) or pa.types.is_list(value_type) or pa.types.is_map(value_type):
                    # Recursively convert nested value type
                    sub_series = pd.Series([value]).where(~pd.Series([value]).isna(), None)
                    sub_array = _convert_to_arrow_array(sub_series, value_type, f"{field_name}.map.value")
                    processed_value = sub_array[0]
                else:
                    processed_value = value
                processed_pairs.append((key, processed_value))
            data.append(processed_pairs)
    
    try:
        return pa.array(data, type=map_type)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{field_name}' to map {map_type}: {str(e)}")


import pandas as pd
import pyarrow as pa
from pyspark.sql.types import Row

# Sample DataFrame with PySpark Row objects
df = pd.DataFrame({
    'id': [1, 2, 3],
    'vendorResult': [
        Row(jumio=None),
        Row(jumio=Row(score=95, details='Verified')),
        None
    ]
})

# Sample PyArrow schema
schema = pa.schema([
    ('id', pa.int64()),
    ('vendorResult', pa.struct([
        ('jumio', pa.struct([
            ('score', pa.int64()),
            ('details', pa.string())
        ]))
    ])),
    ('extra', pa.list_(pa.int64()))  # Missing column
])

# Convert DataFrame to PyArrow Table
table = pandas_to_arrow_with_nested_schema(df, schema)

# Inspect the result
print(table.schema)
print(table.to_pandas())