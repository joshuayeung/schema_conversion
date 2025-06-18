from typing import Optional, Any, Dict, List
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, DataType
from pyspark.sql.functions import lit, col, struct, array, map_from_arrays, when, size, map_keys, expr, explode_outer, collect_list, coalesce

def get_column_paths(schema: StructType, prefix: str = "") -> Dict[str, str]:
    """
    Traverses schema to map full column paths to unique aliases for all fields.
    
    Args:
        schema: Schema to traverse
        prefix: Current path prefix
    
    Returns:
        Dict mapping full paths to unique aliases
    """
    path_to_alias = {}
    for field in schema.fields:
        field_name = f"{prefix}{field.name}" if prefix else field.name
        alias = field_name.replace(".", "_")
        path_to_alias[field_name] = alias
        
        if isinstance(field.dataType, StructType):
            nested_paths = get_column_paths(field.dataType, f"{field_name}.")
            path_to_alias.update(nested_paths)
        elif isinstance(field.dataType, ArrayType):
            current_type = field.dataType
            current_prefix = f"{field_name}."
            while isinstance(current_type, ArrayType):
                if isinstance(current_type.elementType, StructType):
                    nested_paths = get_column_paths(current_type.elementType, current_prefix)
                    path_to_alias.update(nested_paths)
                current_type = current_type.elementType
                current_prefix += "element."
        elif isinstance(field.dataType, MapType) and isinstance(field.dataType.valueType, StructType):
            nested_paths = get_column_paths(field.dataType.valueType, f"{field_name}.value.")
            path_to_alias.update(nested_paths)
    
    return path_to_alias

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
            
            if not subfield_exprs:
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
    For optional structs/maps with required nested fields, sets to NULL if all required fields are NULL or empty.
    For optional arrays, sets to NULL if empty or NULL. For non-nullable arrays, sets to empty array if empty or NULL.
    Recursively normalizes nested arrays and structs at any level using explode and aggregation.
    
    Args:
        df: Input PySpark DataFrame
        schema: Schema (StructType) defining nullability constraints
    
    Returns:
        DataFrame with normalized NULL values for nested structures
    """
    def is_field_empty(col_expr: Any, field_type: Any, prefix: str = "") -> Any:
        """Check if a field is effectively empty (NULL, empty array, or struct with all fields empty)."""
        if isinstance(field_type, StructType):
            conditions = []
            for subfield in field_type.fields:
                subfield_name = f"{prefix}{subfield.name}" if prefix else subfield.name
                subfield_col = col_expr[subfield.name]
                conditions.append(is_field_empty(subfield_col, subfield.dataType, f"{subfield_name}."))
            return coalesce(*conditions, lit(True))
        elif isinstance(field_type, ArrayType):
            return col_expr.isNull() | (size(col_expr) == 0)
        elif isinstance(field_type, MapType):
            return col_expr.isNull() | (size(map_keys(col_expr)) == 0)
        else:
            return col_expr.isNull()
    
    def normalize_struct(current_df: DataFrame, field: StructField, col_expr: Any, prefix: str = "") -> tuple[DataFrame, Any]:
        """Normalize a struct field, setting to NULL if all fields are empty, returning updated DataFrame and expression."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix else field.name
        
        if isinstance(field_type, StructType):
            subfield_exprs = []
            conditions = []
            for subfield in field_type.fields:
                subfield_prefix = f"{field_name}." if field_name else ""
                current_df, subfield_expr = normalize_struct(current_df, subfield, col_expr[subfield.name], subfield_prefix)
                subfield_exprs.append(subfield_expr.alias(subfield.name))
                conditions.append(is_field_empty(col_expr[subfield.name], subfield.dataType, subfield_prefix))
            
            combined_condition = conditions[0] if conditions else lit(True)
            for cond in conditions[1:]:
                combined_condition = combined_condition & cond
            
            if field.nullable:
                struct_expr = when(
                    combined_condition,
                    lit(None)
                ).otherwise(
                    struct(*subfield_exprs)
                ).alias(field_name)
            else:
                struct_expr = struct(*subfield_exprs).alias(field_name)
            
            return current_df, struct_expr
        
        elif isinstance(field_type, ArrayType):
            element_type = field_type.elementType
            temp_col = f"_elem_{field_name.replace('.', '_')}"
            
            # Explode array
            exploded_df = current_df.select(
                explode_outer(col(field_name)).alias(temp_col)
            )
            
            # Normalize array elements
            exploded_df, norm_expr = normalize_struct(
                exploded_df,
                StructField(temp_col, element_type, True),
                col(temp_col),
                ""
            )
            
            exploded_df = exploded_df.withColumn(f"_norm_{temp_col}", norm_expr)
            
            # Aggregate normalized elements
            agg_df = exploded_df.groupBy().agg(
                collect_list(f"_norm_{temp_col}").alias("_temp_array")
            )
            
            # Normalize array based on nullability
            array_expr = col("_temp_array")
            if field.nullable:
                array_expr = when(
                    array_expr.isNull() | (size(array_expr) == 0),
                    lit(None)
                ).otherwise(
                    array_expr
                )
            else:
                array_expr = coalesce(
                    array_expr,
                    array().cast(field_type)
                )
            
            # Rejoin with original DataFrame to preserve other columns
            result_df = current_df.drop(field_name).crossJoin(
                agg_df.select(array_expr.alias(field_name))
            ).drop("_temp_array")
            
            return result_df, col(field_name).alias(field_name)
        
        elif isinstance(field_type, MapType):
            if isinstance(field_type.valueType, StructType):
                value_field = StructField("_val", field_type.valueType, True)
                current_df, value_expr = normalize_struct(current_df, value_field, col("_val"), "")
                map_expr = when(
                    col(field_name).isNotNull(),
                    expr(f"""
                        TRANSFORM_VALUES(
                            COALESCE({field_name}, MAP()),
                            (k, v) -> {value_expr.cast(field_type.valueType).sql_expr()}
                        )
                    """)
                ).otherwise(
                    map_from_arrays(array(), array()).cast(field_type)
                )
            else:
                map_expr = col(field_name)
            
            if field.nullable:
                map_expr = when(
                    col(field_name).isNull() | (size(map_keys(col(field_name))) == 0),
                    lit(None)
                ).otherwise(
                    map_expr
                ).alias(field_name)
            else:
                map_expr = map_expr.alias(field_name)
            
            return current_df, map_expr
        
        else:
            return current_df, col(field_name).alias(field_name)
    
    # Process all fields in a single select
    result_df = df
    field_exprs = []
    for field in schema.fields:
        _, field_expr = normalize_struct(result_df, field, col(field.name), "")
        field_exprs.append(field_expr.alias(field.name))
    
    result_df = result_df.select(*field_exprs)
    
    # Debug: Print final schema and logical plan
    # print("Final result_df schema:")
    # result_df.printSchema()
    # print("Final result_df logical plan:")
    # print(result_df.explain(True))
    
    return result_df
