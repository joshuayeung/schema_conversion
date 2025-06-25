import pandas as pd
import pyarrow as pa
import numpy as np
from pyspark.sql.types import Row
import json
from typing import Any, List, Dict, Tuple

def pandas_to_arrow_with_nested_schema(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """
    Convert a pandas DataFrame to a PyArrow Table, aligning with a schema that may include
    multi-level nested types (struct, list, large_list, map_, dictionary, large_string).
    Missing columns or None values in required fields are filled with default values from
    schema metadata. Supports PySpark Row objects.
    
    Args:
        df (pd.DataFrame): Input pandas DataFrame, may contain Row objects.
        schema (pa.Schema): Target PyArrow schema with nested types and default values
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
    Parse the default value from schema metadata based on the field type, including dictionary,
    large_string, large_list, and map types. Handles nested maps within structs.
    
    Args:
        default_value: Default value from metadata (as bytes).
        field_type: PyArrow data type.
        field_name: Name of the field for error reporting.
    
    Returns:
        Parsed default value compatible with the field type (e.g., list of tuples for maps).
    
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
            parsed = json.loads(default_str)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict for struct field '{field_name}', got {type(parsed)}")
            result = {}
            for subfield in field_type:
                subfield_name = subfield.name
                subfield_type = subfield.type
                subfield_default = subfield.metadata.get(b'default', None) if subfield.metadata else None
                value = parsed.get(subfield_name, None if pa.types.is_map(subfield_type) or subfield.nullable else None)
                if value is not None:
                    if pa.types.is_map(subfield_type) and isinstance(value, list):
                        value = [tuple(item) if isinstance(item, list) and len(item) == 2 else item for item in value]
                    result[subfield_name] = value
                elif subfield_default is not None:
                    result[subfield_name] = _parse_default_value(subfield_default, subfield_type, f"{field_name}.{subfield_name}")
            return result
        elif pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
            return json.loads(default_str)
        elif pa.types.is_map(field_type):
            parsed = json.loads(default_str)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected list of key-value pairs for map field '{field_name}', got {type(parsed)}")
            result = []
            for item in parsed:
                if isinstance(item, list) and len(item) == 2:
                    result.append(tuple(item))
                elif isinstance(item, tuple) and len(item) == 2:
                    result.append(item)
                else:
                    raise ValueError(f"Invalid map default pair {item} in '{field_name}': expected list/tuple of length 2")
            return result
        elif pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
            return default_str
        elif pa.types.is_integer(field_type):
            return int(default_str)
        elif pa.types.is_floating(field_type):
            return float(default_str)
        elif pa.types.is_boolean(field_type):
            return default_str.lower() == 'true'
        else:
            raise ValueError(f"Unsupported field type '{field_type}' for default value in '{field_name}'")
    except Exception as e:
        raise ValueError(f"Failed to parse default value '{default_str}' for field '{field_name}' with type {field_type}: {str(e)}")

def _convert_to_arrow_array(series: pd.Series, field_type: pa.DataType, field_name: str, is_required: bool, default_value: bytes) -> pa.Array:
    """
    Convert a pandas Series to a PyArrow array, handling nested types recursively.
    
    Args:
        series: Pandas Series containing the data.
        field_type: PyArrow data type (may be struct, list, large_list, map_, dictionary, large_string, or primitive).
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
    elif pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
        return _convert_to_list_array(series, field_type, field_name, is_required, default_value)
    elif pa.types.is_map(field_type):
        return _convert_to_map_array(series, field_type, field_name, is_required, default_value)
    else:
        if is_required and default_value is not None:
            default_data = _parse_default_value(default_value, field_type, field_name)
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
                if value is not None:
                    if pa.types.is_map(subfield_type) and isinstance(value, list):
                        if len(value) == 2 and not all(isinstance(i, (list, tuple)) for i in value):
                            value = [tuple(value)]
                        else:
                            value = [tuple(v) if isinstance(v, list) and len(v) == 2 else v for v in value]
                    if pa.types.is_struct(subfield_type) or pa.types.is_list(subfield_type) or pa.types.is_large_list(subfield_type) or pa.types.is_map(subfield_type) or pa.types.is_dictionary(subfield_type):
                        sub_series = pd.Series([value]).where(~pd.Series([value]).isna(), None)
                        sub_array = _convert_to_arrow_array(sub_series, subfield_type, f"{field_name}.{subfield_name}", subfield_required, subfield_default)
                        record[subfield_name] = sub_array[0]
                    else:
                        record[subfield_name] = value
                else:
                    record[subfield_name] = None
            records.append(record)
    
    try:
        return pa.array(records, type=struct_type)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{field_name}' to struct {struct_type}: {str(e)}")

def _convert_to_list_array(series: pd.Series, list_type: pa.ListType, field_name: str, is_required: bool, default_value: bytes) -> pa.Array:
    """
    Convert a pandas Series to a PyArrow list or large_list array, handling nested value types recursively.
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
    value_field = list_type.value_field
    value_default = value_field.metadata.get(b'default', None) if value_field.metadata else None
    value_required = not value_field.nullable
    data = []
    
    default_data = _parse_default_value(default_value, list_type, field_name) if default_value else None
    
    for item in series:
        if item is None:
            if is_required:
                if default_data is None:
                    raise ValueError(f"Required element in '{field_name}' is None and has no default value")
                data.append(default_data)
            else:
                data.append(None)
        elif not isinstance(item, (list, tuple)):
            raise ValueError(f"Expected list or tuple for list field '{field_name}', got {type(item)} in {item}")
        else:
            # Handle non-nullable elements
            processed_items = []
            for element in item:
                if element is None and value_required:
                    if value_default is None:
                        raise ValueError(f"Required element in list field '{field_name}' is None and has no default value")
                    element = _parse_default_value(value_default, value_type, f"{field_name}.list.element")
                processed_items.append(element)
            
            if pa.types.is_struct(value_type) or pa.types.is_list(value_type) or pa.types.is_large_list(value_type) or pa.types.is_map(value_type) or pa.types.is_dictionary(value_type):
                # Recursively convert each element in the list
                if len(processed_items) == 0:
                    data.append([])
                else:
                    sub_series = pd.Series(processed_items).where(~pd.Series(processed_items).isna(), None)
                    sub_array = _convert_to_arrow_array(sub_series, value_type, f"{field_name}.list", value_required, value_default)
                    data.append(sub_array)
            else:
                data.append(processed_items)
    
    try:
        return pa.array(data, type=list_type)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{field_name}' to list {list_type}: {str(e)}")

def _convert_to_map_array(series: pd.Series, map_type: pa.MapType, field_name: str, is_required: bool, default_value: bytes) -> pa.Array:
    """
    Convert a pandas Series to a PyArrow map array, handling nested key/value types recursively.
    Ensures map data is a list of tuples, including single lists like ['key', 'value'].
    """
    series = series.where(~series.isna(), None)
    
    if series.isna().all() or series.isnull().all():
        if is_required:
            if default_value is None:
                raise ValueError(f"Required field '{field_name}' contains all nulls and has no default value")
            default_data = _parse_default_value(default_value, map_type, field_name)
            return pa.array([default_data] * len(series), type=map_type)
        return pa.array([None] * len(series), type=map_type)
    
    key_type = map_type.key_type
    value_field = map_type.item_type
    if isinstance(value_field, pa.DataType):
        value_type = value_field
        value_field = pa.field('item', value_type, True)
        value_default = None
        value_required = False
    else:
        value_type = value_field.type
        value_default = value_field.metadata.get(b'default', None) if value_field.metadata else None
        value_required = not value_field.nullable
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
            if isinstance(item, list) and len(item) == 2 and not all(isinstance(i, (list, tuple)) for i in item):
                item = [tuple(item)]
            if not isinstance(item, (list, tuple)):
                raise ValueError(f"Expected dict, list of tuples, or single key-value list for map field '{field_name}', got {type(item)}: {item}")
            processed_pairs = []
            for pair in item:
                if isinstance(pair, list) and len(pair) == 2:
                    pair = tuple(pair)
                elif not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError(f"Invalid pair {pair} in map field '{field_name}': expected tuple or list of length 2")
                key, value = pair
                if value is None and value_required:
                    if value_default is None:
                        raise ValueError(f"Required map value in '{field_name}' is None and has no default value")
                    value = _parse_default_value(value_default, value_type, f"{field_name}.map.value")
                if pa.types.is_struct(value_type) or pa.types.is_list(value_type) or pa.types.is_large_list(value_type) or pa.types.is_map(value_type) or pa.types.is_dictionary(value_type):
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
        Row(jumio=None, status=None, tags=['Tag1', None], metadata=[('key1', None)]),
        Row(jumio=Row(status='Verified'), status='Approved', tags=['Tag2', 'Tag3'], metadata=[('key2', 'Value2')]),
        None
    ],
    'simple_map': [
        [('k1', None)],
        [('key2', 'v2')],
        None
    ]
})

# Sample schema
schema = pa.schema([
    ('id', pa.int64(), False, {b'default': b'0'}),
    ('vendorResult', pa.struct([
        ('jumio', pa.struct([
            ('status', pa.string(), True)
        ]), True),
        ('status', pa.large_string(), False, {b'default': b'Pending'}),
        ('tags', pa.large_list(pa.field('item', pa.large_string(), False, {b'default': b'Unknown'})), False, {b'default': b'["DefaultTag"]'}),
        ('metadata', pa.map_(pa.string(), pa.field('item', pa.large_string(), False, {b'default': b'Missing'})), False, {b'default': b'[["key", "Missing"]]'}),
    ]), False, {b'default': b'{"jumio": null, "status": "Pending", "tags": ["DefaultTag"], "metadata": [["key", "Missing"]]}'}),
    ('simple_map', pa.map_(pa.string(), pa.large_string()), True),
    ('extra', pa.list_(pa.int64()), True)
])

# Convert DataFrame to PyArrow Table
table = pandas_to_arrow_with_nested_schema(df, schema)

# Inspect the result
print(table.schema)
print(table.to_pandas())
