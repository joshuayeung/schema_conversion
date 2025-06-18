from typing import Optional, Any
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, DataType
from pyspark.sql.functions import lit, col, struct, array, map_from_arrays, when, size, map_keys, explode_outer, collect_list, coalesce

def add_missing_columns(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Adds missing columns and nested subfields to a PySpark DataFrame based on the provided schema.
    
    Args:
        df: Input PySpark DataFrame
        schema: Target schema (StructType)
    
    Returns:
        DataFrame with all columns from the schema
    """
    def get_field_expr(field: StructField, prefix: str = "", existing_field: Optional[StructField] = None) -> Any:
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix else field.name
        
        if isinstance(field_type, StructType):
            existing_subfields = {f.name: f for f in existing_field.dataType.fields} if existing_field and isinstance(existing_field.dataType, StructType) else {}
            subfield_exprs = [
                get_field_expr(subfield, f"{field_name}.", existing_subfields.get(subfield.name)).alias(subfield.name)
                for subfield in field_type.fields
            ]
            return struct(*subfield_exprs).alias(field_name) if subfield_exprs else lit(None).cast(field_type).alias(field_name)
                
        elif isinstance(field_type, ArrayType):
            if existing_field and isinstance(existing_field.dataType, ArrayType):
                if isinstance(field_type.elementType, StructType):
                    existing_element_subfields = {f.name: f for f in existing_field.dataType.elementType.fields} if isinstance(existing_field.dataType.elementType, StructType) else {}
                    subfield_exprs = [
                        col(f"elem.{subfield.name}").alias(subfield.name) if subfield.name in existing_element_subfields
                        else lit(None).cast(subfield.dataType).alias(subfield.name)
                        for subfield in field_type.elementType.fields
                    ]
                    element_expr = struct(*subfield_exprs)
                    return when(
                        col(field_name).isNotNull(),
                        expr(f"""
                            TRANSFORM(COALESCE({field_name}, ARRAY()), elem -> {element_expr})
                        """)
                    ).otherwise(array().cast(field_type)).alias(field_name)
                return col(field_name).alias(field_name)
            return array().cast(field_type).alias(field_name)
            
        elif isinstance(field_type, MapType):
            if existing_field and isinstance(existing_field.dataType, MapType):
                if isinstance(field_type.valueType, StructType):
                    return col(field_name).alias(field_name)  # Preserve existing map
                return col(field_name).alias(field_name)
            return map_from_arrays(array(), array()).cast(field_type).alias(field_name)
            
        else:
            return col(field_name).alias(field_name) if existing_field else lit(None).cast(field_type).alias(field_name)
    
    existing_fields = {f.name: f for f in df.schema.fields}
    select_expr = [get_field_expr(field, "", existing_fields.get(field.name)) for field in schema.fields]
    return df.select(*select_expr)

def normalize_nulls(df: DataFrame, schema: StructType) -> DataFrame:
    """
    Normalizes nullability in a PySpark DataFrame based on the schema.
    
    Args:
        df: Input PySpark DataFrame
        schema: Schema (StructType)
    
    Returns:
        DataFrame with normalized NULL values
    """
    def is_field_empty(col_expr: Any, field_type: Any, prefix: str = "") -> Any:
        if isinstance(field_type, StructType):
            conditions = [
                is_field_empty(col_expr[subfield.name], subfield.dataType, f"{prefix}{subfield.name}.")
                for subfield in field_type.fields
            ]
            return coalesce(*conditions, lit(True))
        elif isinstance(field_type, ArrayType):
            return col_expr.isNull() | (size(col_expr) == 0)
        elif isinstance(field_type, MapType):
            return col_expr.isNull() | (size(map_keys(col_expr)) == 0)
        else:
            return col_expr.isNull()
    
    def normalize_struct(current_df: DataFrame, field: StructField, col_expr: Any, prefix: str = "") -> Any:
        field_type = field.dataType
        field_name = f"{prefix}{field.name}" if prefix else field.name
        
        if isinstance(field_type, StructType):
            subfield_exprs = []
            conditions = []
            for subfield in field_type.fields:
                subfield_expr = normalize_struct(current_df, subfield, col_expr[subfield.name], f"{field_name}.")
                subfield_exprs.append(subfield_expr.alias(subfield.name))
                conditions.append(is_field_empty(col_expr[subfield.name], subfield.dataType, f"{field_name}."))
            
            combined_condition = conditions[0] if conditions else lit(True)
            for cond in conditions[1:]:
                combined_condition = combined_condition & cond
            
            return when(combined_condition, lit(None)).otherwise(struct(*subfield_exprs)).alias(field_name) if field.nullable else struct(*subfield_exprs).alias(field_name)
        
        elif isinstance(field_type, ArrayType):
            element_type = field_type.elementType
            temp_col = f"_elem_{field_name.replace('.', '_')}"
            
            exploded_df = current_df.select(explode_outer(col(field_name)).alias(temp_col))
            element_expr = normalize_struct(exploded_df, StructField(temp_col, element_type, True), col(temp_col), "")
            
            agg_df = exploded_df.withColumn(f"_norm_{temp_col}", element_expr).groupBy().agg(
                collect_list(f"_norm_{temp_col}").alias("_temp_array")
            )
            
            array_expr = col("_temp_array")
            if field.nullable:
                array_expr = when(array_expr.isNull() | (size(array_expr) == 0), lit(None)).otherwise(array_expr)
            else:
                array_expr = coalesce(array_expr, array().cast(field_type))
            
            return current_df.drop(field_name).crossJoin(agg_df.select(array_expr.alias(field_name))).select(col(field_name).alias(field_name))
        
        elif isinstance(field_type, MapType):
            if isinstance(field_type.valueType, StructType):
                value_field = StructField("value", field_type.valueType, True)
                value_expr = normalize_struct(current_df, value_field, col("value"), "")
                map_expr = when(
                    col(field_name).isNotNull(),
                    expr(f"""
                        TRANSFORM_VALUES(COALESCE({field_name}, MAP()), (k, v) -> {value_expr})
                    """)
                ).otherwise(map_from_arrays(array(), array()).cast(field_type))
            else:
                map_expr = col(field_name)
            
            return when(col(field_name).isNull() | (size(map_keys(col(field_name))) == 0), lit(None)).otherwise(map_expr).alias(field_name) if field.nullable else map_expr.alias(field_name)
        
        else:
            return col(field_name).alias(field_name)
    
    field_exprs = [normalize_struct(df, field, col(field.name), "").alias(field.name) for field in schema.fields]
    return df.select(*field_exprs)
