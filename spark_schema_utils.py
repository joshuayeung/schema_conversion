from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, NullType
from pyspark.sql.functions import lit, col, when, array, struct, map_from_arrays, coalesce, expr, size

def add_missing_columns(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Adds missing columns and nested subfields to a PySpark DataFrame based on the provided schema.
    Missing columns and subfields (including nested structs, maps, arrays) are added as NULL.
    
    Args:
        df: Input PySpark DataFrame
        schema: Target schema (StructType) to align the DataFrame with
    
    Returns:
        DataFrame with all columns and nested subfields from the schema, missing ones added as NULL
    """
    def get_field_expr(field: StructField, prefix: str = "", existing_field: StructField = None, return_sql: bool = False) -> tuple:
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
                if existing_subfield and isinstance(existing_field, StructField):
                    subfield_expr = col(f"{field_name}.{subfield.name}")
                    subfield_sql = f"{field_name}.{subfield.name}"
                    if isinstance(subfield.dataType, (StructType, ArrayType, MapType)):
                        nested_expr, nested_sql = get_field_expr(
                            subfield,
                            subfield_prefix,
                            existing_subfield,
                            return_sql=True
                        )
                        subfield_expr = nested_expr
                        subfield_sql = nested_sql
                else:
                    subfield_expr, subfield_sql = get_field_expr(
                        subfield,
                        subfield_prefix,
                        None,
                        return_sql=True
                    )
                subfield_expr = subfield_expr.alias(subfield.name)
                subfield_sqls.append(f"{subfield_sql} AS {subfield.name}")
            
            if return_sql:
                return struct(*subfield_exprs), f"STRUCT({', '.join(subfield_sqls)})"
            return struct(*subfield_exprs).alias(field_name)
                
        elif isinstance(field_type, ArrayType):
            if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, ArrayType):
                if isinstance(field_type.elementType, StructType):
                    element_field = StructField("elem", field_type.elementType, True)  # Unique alias to avoid conflicts
                    existing_element_field = StructField("elem", existing_field.dataType.elementType, True) if isinstance(existing_field.dataType.elementType, StructType) else None
                    _, element_sql = get_field_expr(element_field, "elem.", existing_element_field, return_sql=True)
                    array_expr = when(
                        col(field_name).isNotNull(),
                        expr(f"""
                            (SELECT ARRAY_AGG({element_sql})
                             FROM (SELECT EXPLODE(COALESCE({field_name}, ARRAY())) AS elem))
                        """)
                    ).otherwise(
                        array().cast(field_type)
                    ).alias(field_name)
                    if return_sql:
                        return array_expr, f"""
                            CASE WHEN {field_name} IS NOT NULL
                                 THEN (SELECT ARRAY_AGG({element_sql})
                                       FROM (SELECT EXPLODE(COALESCE({field_name}, ARRAY())) AS elem))
                                 ELSE ARRAY()
                            END
                        """
                    return array_expr
                array_expr = col(field_name).alias(field_name)
                if return_sql:
                    return array_expr, field_name
                return array_expr
            array_expr = array().cast(field_type).alias(field_name)
            if return_sql:
                return array_expr, "ARRAY()"
            return array_expr
            
        elif isinstance(field_type, MapType):
            if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, MapType):
                if isinstance(field_type.valueType, StructType):
                    value_field = StructField("val", field_type.valueType, True)  # Unique alias to avoid conflicts
                    existing_value_field = StructField("val", existing_field.dataType.valueType, True) if isinstance(existing_field.dataType.valueType, StructType) else None
                    _, value_sql = get_field_expr(value_field, "val.", existing_value_field, return_sql=True)
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
                    ).alias(field_name)
                    if return_sql:
                        return map_expr, f"""
                            CASE WHEN {field_name} IS NOT NULL
                                 THEN TRANSFORM_VALUES(
                                      COALESCE({field_name}, MAP()),
                                      (k, v) -> {value_sql}
                                 )
                                 ELSE MAP()
                            END
                        """
                    return map_expr
                map_expr = col(field_name).alias(field_name)
                if return_sql:
                    return map_expr, field_name
                return map_expr
            map_expr = map_from_arrays(array(), array()).cast(field_type).alias(field_name)
            if return_sql:
                return map_expr, "MAP()"
            return map_expr
            
        else:
            if isinstance(existing_field, StructField):
                column_expr = col(field_name).alias(field_name)
                sql_expr = field_name
            else:
                column_expr = lit(None).cast(field_type).alias(field_name)
                sql_expr = f"CAST(NULL AS {field_type.simpleString()})"
            if return_sql:
                return column_expr, sql_expr
            return column_expr
    
    existing_columns = {f.name: f for f in df.schema.fields}
    select_expr = []
    
    for field in schema.fields:
        existing_field = existing_columns.get(field.name)
        select_expr.append(get_field_expr(field, "", existing_field))
    
    return df.select(*select_expr)

def normalize_nulls(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Normalizes nullability in a PySpark DataFrame based on the schema.
    For optional structs/maps/arrays with required nested fields, sets the entire
    field to NULL if all required nested fields are NULL.
    
    Args:
        df: Input PySpark DataFrame
        schema: Schema (StructType) defining nullability constraints
    
    Returns:
        DataFrame with normalized NULL values for nested structures
    """
    def build_null_condition(field: StructField, prefix: str = "", existing_field: StructField = None) -> str:
        """Build condition to check if a field should be set to NULL."""
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix else field.name
        
        if isinstance(field_type, StructType):
            existing_struct_type = existing_field.dataType if isinstance(existing_field, StructField) and isinstance(existing_field.dataType, StructType) else None
            existing_subfields = {f.name: f for f in existing_struct_type.fields} if existing_struct_type else {}
            
            required_fields = [f for f in field_type.fields if not f.nullable]
            if not required_fields:
                return col(field_name).alias(field_name)
            
            subfield_exprs = []
            conditions = []
            for subfield in field_type.fields:
                existing_subfield = existing_subfields.get(subfield.name)
                subfield_expr = build_null_condition(
                    subfield,
                    f"{field_name}." if field_name else "",
                    existing_subfield
                )
                subfield_expr = subfield_expr.alias(subfield.name)
                subfield_exprs.append(subfield_expr)
                if not subfield.nullable:
                    conditions.append(col(f"{field_name}.{subfield.name}").isNull())
            
            combined_condition = conditions[0] if conditions else lit(True)
            for cond in conditions[1:]:
                combined_condition = combined_condition & cond
            
            if field.nullable:
                return when(
                    combined_condition,
                    lit(None)
                ).otherwise(
                    struct(*subfield_exprs)
                ).alias(field_name)
            return struct(*subfield_exprs).alias(field_name)
                
        elif isinstance(field_type, ArrayType):
            if field.nullable:
                return when(
                    col(field_name).isNull() | (size(col(field_name)) == 0),
                    lit(None)
                ).otherwise(
                    col(field_name)
                ).alias(field_name)
            return col(field_name).alias(field_name)
            
        elif isinstance(field_type, MapType):
            if field.nullable:
                return when(
                    col(field_name).isNull() | (size(map_keys(col(field_name))) == 0),
                    lit(None)
                ).otherwise(
                    col(field_name)
                ).alias(field_name)
            return col(field_name).alias(field_name)
            
        else:
            return col(field_name).alias(field_name)
    
    existing_columns = {f.name: f for f in df.schema.fields}
    select_expr = []
    
    for field in schema.fields:
        existing_field = existing_columns.get(field.name)
        select_expr.append(build_null_condition(field, "", existing_field))
    
    return df.select(*select_expr)
