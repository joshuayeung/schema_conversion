from typing import Optional, Any
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType
from pyspark.sql.functions import lit, col, struct, array, map_from_arrays, when, coalesce, expr, size, filter, transform, transform_values, map_keys

def add_missing_columns(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Adds missing columns and nested subfields to a PySpark DataFrame based on the provided schema.
    Missing columns and subfields (including nested structs, arrays, maps) are added as NULL.
    
    Args:
        df: Input PySpark DataFrame
        schema: Target schema (StructType) to align the DataFrame with
    
    Returns:
        DataFrame with all columns and nested subfields from the schema, missing ones added as NULL
    """
    def get_field_expr(field: StructField, prefix: str = "", existing_field: Optional[StructField] = None, return_sql: bool = False) -> Any:
        """Generate expression for a field, handling nested structures and existing fields."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix else field.name
        
        if isinstance(field_type, StructType):
            existing_struct_type = existing_field.dataType if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, StructType) else None
            existing_subfields = {f.name: f for f in existing_struct_type.fields} if existing_struct_type else {}
            
            subfield_exprs = []
            subfield_sqls = []
            for subfield in field_type.fields:
                existing_subfield = existing_subfields.get(subfield.name)
                subfield_prefix = f"{field_name}." if field_name else ""
                nested_expr, nested_sql = get_field_expr(
                    subfield,
                    subfield_prefix,
                    existing_subfield,
                    return_sql=True
                )
                subfield_expr = nested_expr
                subfield_sql = nested_sql
                subfield_expr = subfield_expr.alias(subfield.name)
                subfield_exprs.append(subfield_expr)
                subfield_sqls.append(f"{subfield_sql} AS {subfield.name}")
            
            if not subfield_exprs:  # Handle empty struct
                if isinstance(existing_field, StructField):
                    col_expr = col(field_name)
                    sql = field_name
                else:
                    col_expr = lit(None).cast(field_type)
                    sql = f"CAST(NULL AS {field_type.simpleString()})"
                if return_sql:
                    return col_expr, sql
                return col_expr.alias(field_name)
            
            col_expr = struct(*subfield_exprs)
            sql = f"STRUCT({', '.join(subfield_sqls)})"
            if return_sql:
                return col_expr, sql
            return col_expr.alias(field_name)
                
        elif isinstance(field_type, ArrayType):
            if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, ArrayType):
                if isinstance(field_type.elementType, StructType):
                    element_field = StructField("_elem", field_type.elementType, True)
                    existing_element_field = StructField("_elem", existing_field.dataType.elementType, True) if isinstance(existing_field.dataType.elementType, StructType) else None
                    element_struct_type = field_type.elementType
                    existing_element_struct = existing_element_field.dataType if existing_element_field else None
                    existing_element_subfields = {f.name: f for f in existing_element_struct.fields} if existing_element_struct else {}
                    element_sql_parts = []
                    element_exprs = []
                    for subfield in element_struct_type.fields:
                        if subfield.name in existing_element_subfields:
                            element_sql_parts.append(f"_elem.{subfield.name} AS {subfield.name}")
                            element_expr = col(f"_elem.{subfield.name}")
                        else:
                            subfield_expr, subfield_sql = get_field_expr(subfield, "", None, return_sql=True)
                            element_sql_parts.append(f"{subfield_sql} AS {subfield.name}")
                            element_expr = subfield_expr
                        element_exprs.append(element_expr.alias(subfield.name))
                    element_sql = f"STRUCT({', '.join(element_sql_parts)})"
                    element_expr = struct(*element_exprs)
                    array_expr = when(
                        col(field_name).isNotNull(),
                        expr(f"""
                            (SELECT ARRAY_AGG({element_sql})
                             FROM (SELECT EXPLODE(COALESCE({field_name}, ARRAY())) AS _elem))
                        """)
                    ).otherwise(
                        array().cast(field_type)
                    )
                    array_sql = f"""
                        CASE WHEN {field_name} IS NOT NULL
                             THEN (SELECT ARRAY_AGG({element_sql})
                                   FROM (SELECT EXPLODE(COALESCE({field_name}, ARRAY())) AS _elem))
                             ELSE ARRAY()
                        END
                    """
                    if return_sql:
                        return array_expr, array_sql
                    return array_expr.alias(field_name)
                col_expr = col(field_name)
                sql = field_name
                if return_sql:
                    return col_expr, sql
                return col_expr.alias(field_name)
            col_expr = array().cast(field_type)
            sql = "ARRAY()"
            if return_sql:
                return col_expr, sql
            return col_expr.alias(field_name)
            
        elif isinstance(field_type, MapType):
            if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, MapType):
                if isinstance(field_type.valueType, StructType):
                    value_field = StructField("_val", field_type.valueType, True)
                    existing_value_field = StructField("_val", existing_field.dataType.valueType, True) if isinstance(existing_field.dataType.valueType, StructType) else None
                    value_expr, value_sql = get_field_expr(value_field, "", existing_value_field, return_sql=True)
                    map_expr = when(
                        col(field_name).isNotNull(),
                        expr(f"""
                            TRANSFORM_VALUES(
                                COALESCE({field_name}, MAP()),
                                (k, v) -> {value_sql}
                            )
                        """)
                    ).otherwise(
                        map_from_arrays(array(), array()).cast(field_type)
                    )
                    map_sql = f"""
                        CASE WHEN {field_name} IS NOT NULL
                             THEN TRANSFORM_VALUES(
                                  COALESCE({field_name}, MAP()),
                                  (k, v) -> {value_sql}
                             )
                             ELSE MAP()
                        END
                    """
                    if return_sql:
                        return map_expr, map_sql
                    return map_expr.alias(field_name)
                col_expr = col(field_name)
                sql = field_name
                if return_sql:
                    return col_expr, sql
                return col_expr.alias(field_name)
            col_expr = map_from_arrays(array(), array()).cast(field_type)
            sql = "MAP()"
            if return_sql:
                return col_expr, sql
            return col_expr.alias(field_name)
            
        else:
            if isinstance(existing_field, StructField):
                col_expr = col(field_name)
                sql = field_name
            else:
                col_expr = lit(None).cast(field_type)
                sql = f"CAST(NULL AS {field_type.simpleString()})"
            if return_sql:
                return col_expr, sql
            return col_expr.alias(field_name)
    
    existing_columns = {f.name: f for f in df.schema.fields}
    select_expr = []
    
    for field in schema.fields:
        existing_field = existing_columns.get(field.name)
        select_expr.append(get_field_expr(field, "", existing_field))
    
    return df.select(*select_expr)

def normalize_nulls(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Normalizes nullability in a PySpark DataFrame based on the schema.
    For optional structs/maps with required nested fields, sets the entire field to NULL if all required fields are NULL.
    For optional arrays, sets to NULL if empty or NULL. For non-nullable arrays, sets to NULL if empty, NULL, or all required nested fields are NULL.
    Recursively normalizes nested arrays and structs at any level.
    
    Args:
        df: Input PySpark DataFrame
        schema: Schema (StructType) defining nullability constraints
    
    Returns:
        DataFrame with normalized NULL values for nested structures
    """
    def build_null_condition(field: StructField, prefix: str = "", existing_field: Optional[StructField] = None, is_array_element: bool = False) -> Any:
        """Build condition to check if a field should be set to NULL, recursively handling nested structures."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix and not is_array_element else field.name
        
        if isinstance(field_type, StructType):
            existing_struct_type = existing_field.dataType if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, StructType) else None
            existing_subfields = {f.name: f for f in existing_struct_type.fields} if existing_struct_type else {}
            
            required_fields = [f for f in field_type.fields if not f.nullable]
            subfield_exprs = []
            conditions = []
            
            for subfield in field_type.fields:
                existing_subfield = existing_subfields.get(subfield.name)
                subfield_prefix = f"{field_name}." if field_name and not is_array_element else ""
                subfield_expr = build_null_condition(
                    subfield,
                    subfield_prefix,
                    existing_subfield,
                    is_array_element=False
                )
                subfield_expr = subfield_expr.alias(subfield.name)
                subfield_exprs.append(subfield_expr)
                if not subfield.nullable:
                    conditions.append(col(f"{subfield_prefix}{subfield.name}").isNull())
            
            if not required_fields:
                return col(field_name).alias(field_name) if field_name and not is_array_element else struct(*subfield_exprs)
            
            combined_condition = conditions[0] if conditions else lit(True)
            for cond in conditions[1:]:
                combined_condition = combined_condition & cond
            
            if field.nullable:
                return when(
                    combined_condition,
                    lit(None)
                ).otherwise(
                    struct(*subfield_exprs)
                ).alias(field_name) if field_name and not is_array_element else when(
                    combined_condition,
                    lit(None)
                ).otherwise(struct(*subfield_exprs))
            return struct(*subfield_exprs).alias(field_name) if field_name and not is_array_element else struct(*subfield_exprs)
                
        elif isinstance(field_type, ArrayType):
            col_ref = col(field_name) if not is_array_element else col("x")
            
            # Handle array elements recursively if they are structs
            if isinstance(field_type.elementType, StructType):
                element_struct_type = field_type.elementType
                element_fields = []
                for subfield in element_struct_type.fields:
                    subfield_result = build_null_condition(
                        subfield,
                        prefix="",
                        existing_field=subfield,
                        is_array_element=True
                    )
                    element_fields.append((subfield.name, subfield_result))
                transformed_elements = transform(
                    col_ref,
                    lambda x: struct(*[x[field_name] if isinstance(field_result, str) else field_result for field_name, field_result in element_fields])
                )
            else:
                transformed_elements = col_ref
            
            if field.nullable:
                # Optional arrays: set to NULL if NULL or empty
                return when(
                    col_ref.isNull() | (size(col_ref) == 0),
                    lit(None)
                ).otherwise(
                    transformed_elements
                ).alias(field_name) if not is_array_element else when(
                    col_ref.isNull() | (size(col_ref) == 0),
                    lit(None)
                ).otherwise(
                    transformed_elements
                )
            
            else:
                # Non-nullable arrays: check if empty or all required fields are NULL
                required_fields = []
                if isinstance(field_type.elementType, StructType):
                    required_fields = [f for f in field_type.elementType.fields if not f.nullable]
                
                if required_fields:
                    conditions = []
                    for req_field in required_fields:
                        non_null_elements = filter(
                            col_ref,
                            lambda x: x[req_field.name].isNotNull()
                        )
                        conditions.append(size(non_null_elements) == 0)
                    
                    combined_condition = conditions[0] if conditions else lit(True)
                    for cond in conditions[1:]:
                        combined_condition = combined_condition & cond
                    
                    return when(
                        col_ref.isNull() | (size(col_ref) == 0) | combined_condition,
                        lit(None)
                    ).otherwise(
                        transformed_elements
                    ).alias(field_name) if not is_array_element else when(
                        col_ref.isNull() | (size(col_ref) == 0) | combined_condition,
                        lit(None)
                    ).otherwise(
                        transformed_elements
                    )
                
                # Non-nullable arrays without required fields: NULL if empty
                return when(
                    col_ref.isNull() | (size(col_ref) == 0),
                    lit(None)
                ).otherwise(
                    transformed_elements
                ).alias(field_name) if not is_array_element else when(
                    col_ref.isNull() | (size(col_ref) == 0),
                    lit(None)
                ).otherwise(
                    transformed_elements
                )
            
        elif isinstance(field_type, MapType):
            if isinstance(field_type.valueType, StructType):
                value_field = StructField("_val", field_type.valueType, True)
                value_expr = build_null_condition(value_field, prefix="")
                map_expr = when(
                    col(field_name).isNotNull(),
                    transform_values(
                        col(field_name),
                        lambda k, v: value_expr
                    )
                ).otherwise(
                    map_from_arrays(array(), array()).cast(field_type)
                )
            else:
                map_expr = col(field_name)
            
            if field.nullable:
                return when(
                    col(field_name).isNull() | (size(map_keys(col(field_name))) == 0),
                    lit(None)
                ).otherwise(
                    map_expr
                ).alias(field_name) if not is_array_element else when(
                    col(field_name).isNull() | (size(map_keys(col(field_name))) == 0),
                    lit(None)
                ).otherwise(
                    map_expr
                )
            return map_expr.alias(field_name) if not is_array_element else map_expr
            
        else:
            # For array element fields, return field name to be used in lambda
            if is_array_element:
                return field_name
            return col(field_name).alias(field_name) if field_name and not is_array_element else col(field_name)
    
    existing_columns = {f.name: f for f in df.schema.fields}
    select_expr = []
    
    for field in schema.fields:
        existing_field = existing_columns.get(field.name)
        select_expr.append(build_null_condition(field, "", existing_field))
    
    return df.select(*select_expr)
