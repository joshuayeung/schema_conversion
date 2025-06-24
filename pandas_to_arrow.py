import pandas as pd
import pyarrow as pa
import numpy as np
from pyspark.sql.types import Row
import json
from typing import Any, List, Dict

def pandas_to_arrow_with_nested_schema(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """
    Convert a pandas DataFrame to a PyArrow Table, aligning with a schema that may include
    multi-level nested types (struct, list, map_, dictionary, large_string). Missing columns
    or None values in required fields are filled with default values from schema metadata.
    Supports PySpark Row objects.
    
    Args:
        df (pd.DataFrame): Input pandas DataFrame, possibly containing Row objects.
        schema (pa.Schema): Target PyArrow schema with possible nested types and default values
                           in metadata (e.g., {'default': 'value'}).
    
    Returns:
        pa.Table: PyArrow Table conforming to the provided schema.
    
    Raises:
        ValueError: If DataFrame contains columns not in schema, data is incompatible, or
                    required fields lack default values when needed.
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
        is_required = not field.nullable
        default_value = field.metadata.get(b'default', None) if field.metadata else None
        
        if field_name in df:
            # Column exists in DataFrame, convert to PyArrow array
            column_data = df[field_name]
            array = _convert_to_arrow_array(column_data, field_type, field_name, is_required, default_value)
        else:
            # Column missing
            if is_required:
                if default_value is None:
                    raise ValueError(f"Required field '{field_name}' is missing and has no default value")
                # Create array with default value
                default_data = _parse_default_value(default_value, field_type, field_name)
                array = pa.array([default_data] * len(df), type=field_type)
            else:
                # Create null array for non-required missing fields
                array = pa.array([None] * len(df), type=field_type)
        
        arrays.append(array)
    
    # Create PyArrow Table from arrays
    return pa.Table.from_arrays(arrays, schema=schema)

def _parse_default_value(default_value: bytes, field_type: pa.DataType, field_name: str) -> Any:
    """
    Parse the default value from schema metadata based on the field type, including dictionary and large_string types.
    
    Args:
        default_value: Default value from metadata (as bytes).
        field_type: PyArrow data type.
        field_name: Name of the field for error reporting.
    
    Returns:
        Parsed default value compatible with the field type.
    
    Raises:
        ValueError: If default value is invalid or incompatible with the field type.
    """
    if default_value is None:
        raise ValueError(f"No default value provided for required field '{field_name}'")
    
    # Decode bytes to string
    default_str = default_value.decode('utf-8')
    
    try:
        if pa.types.is_dictionary(field_type):
            # Handle dictionary (enum) type; default must match value_type
            value_type = field_type.value_type
            if pa.types.is_string(value_type) or pa.types.is_large_string(value_type):
                return default_str
            elif pa.types.is_integer(value_type):
                return int(default_str)
            elif pa.types.is_floating(value_type):
                return float(default_str)
            else:
                raise ValueError(f"Unsupported dictionary value type '{value_type}' for field '{field_name}'")
        elif pa.types.is_struct(field_type):
            return json.loads(default_str)  # Expect JSON string for structs
        elif pa.types.is_list(field_type):
            return json.loads(default_str)  # Expect JSON string for lists
        elif pa.types.is_map(field_type):
            return json.loads(default_str)  # Expect JSON string for maps (list of [key, value])
        elif pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
            return default_str
        elif pa.types.is_integer(field_type):
            return int(default_str)
        elif pa.types.is_floating(field_type):
            return float(default_str)
        elif pa.types.is_boolean(field_type):
            return default_str.lower() == 'true'
        else:
            raise ValueError(f"Unsupported field type '{field_type}' for default value in field '{field_name}'")
    except Exception as e:
        raise ValueError(f"Failed to parse default value '{default_str}' for field '{field_name}' with type {field_type}: {str(e)}")

def _convert_to_arrow_array(series: pd.Series, field_type: pa.DataType, field_name: str, is_required: bool, default_value: bytes) -> pa.Array:
    """
    Convert a pandas Series to a PyArrow array, handling nested types recursively.
    
    Args:
        series: Pandas Series containing the data.
        field_type: PyArrow data type (may be struct, list, map_, dictionary, large_string, or primitive).
        field_name: Name of the field for error reporting.
        is_required: Whether the field is required (non-nullable).
        default_value: Default value from schema metadata (as bytes).
    
    Returns:
        pa.Array: PyArrow array matching the field type.
    
    Raises:
        ValueError: If data is incompatible with the field type or required field lacks default.
    """
    # Replace pandas NA/NaN with None
    series = series.where(~series.isna(), None)
    
    if series.isna().all() or series.isnull().all():
        if is_required:
            if default_value is None:
                raise ValueError(f"Required field '{field_name}' contains all nulls and has no default value")
            default_data = _parse_default_value(default_value, field_type, field_name)
            return pa.array([default_data] * len(series), type=field_type)
        return pa.array([None] * len(series), type=field_type)
    
    if pa.types.is_struct(field_type):
        return _convert_to_struct_array(series, field_type, field_name, is_required, default_value)
    elif pa.types.is_list(field_type):
        return _convert_to_list_array(series, field_type, field_name, is_required, default_value)
    elif pa.types.is_map(field_type):
        return _convert_to_map_array(series, field_type, field_name, is_required, default_value)
    else:
        # Handle primitive, dictionary, and large_string types
        if is_required:
            default_data = _parse_default_value(default_value, field_type, field_name) if default_value else None
            if default_data is not None:
                series = series.where(series.notna(), default_data)
        try:
            return pa.array(series, type=field_type)
        except Exception as e:
            raise ValueError(f"Failed to convert column '{field_name}' to {field_type}: {str(e)}")

def _convert_to_struct_array(series: pd.Series, struct_type: pa.StructType, field_name: str, is_required: bool, default_value: bytes) -> pa.Array:
    """
    Convert a pandas Series to a PyArrow struct array, handling nested fields recursively.
    Supports PySpark Row objects and dictionaries.
    """
    # Replace pandas NA/NaN with None
    series = series.where(~series.isna(), None)
    
    if series.isna().all() or series.isnull().all():
        if is_required:
            if default_value is None:
                raise ValueError(f"Required field '{field_name}' contains all nulls and has no default value")
            default_data = _parse_default_value(default_value, struct_type, field_name)
            return pa.array([default_data] * len(series), type=struct_type)
        return pa.array([None] * len(series), type=struct_type)
    
    # Extract struct fields
    struct_fields = {f.name: f.type for f in struct_type}
    records = []
    
    default_data = _parse_default_value(default_value, struct_type, field_name) if default_value else None
    
    for item in series:
        if item is None:
            if is_required:
                if default_data is None:
                    raise ValueError(f"Required field '{field_name}' contains None and has no default value")
                records.append(default_data)
            else:
                records.append(None)
        else:
            # Convert PySpark Row to dict if necessary
            if isinstance(item, Row):
                item = item.asDict(recursive=True)
            if not isinstance(item, dict):
                raise ValueError(f"Expected dict or Row for struct field '{field_name}', got {type(item)} in {item}")
            record = {}
            for subfield_name, subfield_type in struct_fields.items():
                # Get subfield metadata
                subfield = struct_type.field(subfield_name)
                subfield_default = subfield.metadata.get(b'default', None) if subfield.metadata else None
                subfield_required = not subfield.nullable
                value = item.get(subfield_name, None)
                if value is None and subfield_required:
                    if subfield_default is None:
                        raise ValueError(f"Required subfield '{field_name}.{subfield_name}' is None and has no default value")
                    value = _parse_default_value(subfield_default, subfield_type, f"{field_name}.{subfield_name}")
                if value is not None and (pa.types.is_struct(subfield_type) or pa.types.is_list(subfield_type) or pa.types.is_map(subfield_type) or pa.types.is_dictionary(subfield_type)):
                    # Recursively convert nested types and dictionary types
                    sub_series = pd.Series([value]).where(~pd.Series([value]).isna(), None)
                    sub_array = _convert_to_arrow_array(sub_series, subfield_type, f"{field_name}.{subfield_name}", subfield_required, subfield_default)
                    record[subfield_name] = sub_array[0]
                else:
                    record[subfield_name] = value
            records.append(record)
    
    try:
        return pa.array(records, type=struct_type)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{field_name}' to struct {struct_type}: {str(e)}")

def _convert_to_list_array(series: pd.Series, list_type: pa.ListType, field_name: str, is_required: bool, default_value: bytes) -> pa.Array:
    """
    Convert a pandas Series to a PyArrow list array, handling nested value types recursively.
    """
    # Replace pandas NA/NaN with None
    series = series.where(~series.isna(), None)
    
    if series.isna().all() or series.isnull().all():
        if is_required:
            if default_value is None:
                raise ValueError(f"Required field '{field_name}' contains all nulls and has no default value")
            default_data = _parse_default_value(default_value, list_type, field_name)
            return pa.array([default_data] * len(series), type=list_type)
        return pa.array([None] * len(series), type=list_type)
    
    value_type = list_type.value_type
    value_field = pa.field("item", value_type, value_type.nullable)
    value_default = value_field.metadata.get(b'default', None) if value_field.metadata else None
    value_required = not value_type.nullable
    data = []
    
    default_data = _parse_default_value(default_value, list_type, field_name) if default_value else None
    
    for item in series:
        if item is None:
            if is_required:
                if default_data is None:
                    raise ValueError(f"Required field '{field_name}' contains None and has no default value")
                data.append(default_data)
            else:
                data.append(None)
        elif not isinstance(item, (list, tuple)):
            raise ValueError(f"Expected list or tuple for list field '{field_name}', got {type(item)} in {item}")
        else:
            if pa.types.is_struct(value_type) or pa.types.is_list(value_type) or pa.types.is_map(value_type) or pa.types.is_dictionary(value_type):
                # Recursively convert each element in the list
                if len(item) == 0:
                    data.append([])
                else:
                    sub_series = pd.Series(item).where(~pd.Series(item).isna(), None)
                    sub_array = _convert_to_arrow_array(sub_series, value_type, f"{field_name}.list", value_required, value_default)
                    data.append(sub_array)
            else:
                data.append(item)
    
    try:
        return pa.array(data, type=list_type)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{field_name}' to list {list_type}: {str(e)}")

def _convert_to_map_array(series: pd.Series, map_type: pa.MapType, field_name: str, is_required: bool, default_value: bytes) -> pa.Array:
    """
    Convert a pandas Series to a PyArrow map array, handling nested key/value types recursively.
    """
    # Replace pandas NA/NaN with None
    series = series.where(~series.isna(), None)
    
    if series.isna().all() or series.isnull().all():
        if is_required:
            if default_value is None:
                raise ValueError(f"Required field '{field_name}' contains all nulls and has no default value")
            default_data = _parse_default_value(default_value, map_type, field_name)
            return pa.array([default_data] * len(series), type=map_type)
        return pa.array([None] * len(series), type=map_type)
    
    key_type = map_type.key_type
    value_type = map_type.item_type
    value_field = pa.field("item", value_type, value_type.nullable)
    value_default = value_field.metadata.get(b'default', None) if value_field.metadata else None
    value_required = not value_type.nullable
    data = []
    
    default_data = _parse_default_value(default_value, map_type, field_name) if default_value else None
    
    for item in series:
        if item is None:
            if is_required:
                if default_data is None:
                    raise ValueError(f"Required field '{field_name}' contains None and has no default value")
                data.append(default_data)
            else:
                data.append(None)
        else:
            if isinstance(item, dict):
                item = list(item.items())
            if not isinstance(item, (list, tuple)):
                raise ValueError(f"Expected dict or list of tuples for map field '{field_name}', got {type(item)} in {item}")
            
            # Process key-value pairs
            processed_pairs = []
            for key, value in item:
                if value is None and value_required:
                    if value_default is None:
                        raise ValueError(f"Required map value in '{field_name}' is None and has no default value")
                    value = _parse_default_value(value_default, value_type, f"{field_name}.map.value")
                if pa.types.is_struct(value_type) or pa.types.is_list(value_type) or pa.types.is_map(value_type) or pa.types.is_dictionary(value_type):
                    # Recursively convert nested value type
                    sub_series = pd.Series([value]).where(~pd.Series([value]).isna(), None)
                    sub_array = _convert_to_arrow_array(sub_series, value_type, f"{field_name}.map.value", value_required, value_default)
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

# Sample DataFrame
df = pd.DataFrame({
    'id': [1, 2, 3],
    'vendorResult': [
        Row(jumio=None, status=None),  # status should use default 'Pending'
        Row(jumio=Row(score=95, details='Verified'), status='Approved'),
        None  # vendorResult should use default
    ]
})

# Sample schema with optional jumio and required large_string status
schema = pa.schema([
    ('id', pa.int64(), False, {b'default': b'0'}),  # Required with default 0
    ('vendorResult', pa.struct([
        ('jumio', pa.struct([
            ('score', pa.int64(), True),  # Optional
            ('details', pa.string(), True)  # Optional
        ]), True),  # Optional
        ('status', pa.large_string(), False, {b'default': b'Pending'}),  # Required with default 'Pending'
        ('category', pa.dictionary(pa.int64(), pa.string()), False, {b'default': b'Unknown'})  # Required dictionary
    ]), False, {b'default': b'{"jumio": null, "status": "Pending", "category": "Unknown"}'}),  # Required with default
    ('extra', pa.list_(pa.int64()), True)  # Optional, no default
])

# Convert DataFrame to PyArrow Table
table = pandas_to_arrow_with_nested_schema(df, schema)

# Inspect the result
print(table.schema)
print(table.to_pandas())
