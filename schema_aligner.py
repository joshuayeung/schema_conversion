from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, DataType
from pyspark.sql.functions import lit, col, struct, array, when, isnull
def align_schema(df, target_schema):
    """
    Aligns a PySpark DataFrame's schema to a target schema by adding missing fields with NULL values.
    Handles nested structures (structs, arrays, maps) recursively.
    
    Args:
        df: Input PySpark DataFrame
        target_schema: Target schema (StructType) to align with
    
    Returns:
        DataFrame with schema aligned to target_schema
    """
    def get_nested_fields(schema, prefix=""):
        """Helper to get all field names in schema with dot notation for nested fields."""
        fields = []
        for field in schema:
            full_name = f"{prefix}{field.name}" if prefix else field.name
            if isinstance(field.dataType, StructType):
                fields.extend(get_nested_fields(field.dataType, f"{full_name}."))
            else:
                fields.append(full_name)
        return fields
    
    def align_struct(df, curr_schema, target_schema, prefix=""):
        """Recursively aligns a struct schema."""
        output_fields = []
        for field in target_schema:
            full_name = f"{prefix}{field.name}" if prefix else field.name
            # Check if field exists in current schema
            curr_field = next((f for f in curr_schema if f.name == field.name), None)
            
            if not curr_field:
                # Field doesn't exist, add it with NULL
                output_fields.append(lit(None).cast(field.dataType).alias(field.name))
            else:
                # Field exists, check type
                if isinstance(field.dataType, StructType):
                    # Handle nested struct
                    nested_df = align_struct(df, curr_field.dataType, field.dataType, f"{full_name}.")
                    output_fields.append(struct(*[col(f"{full_name}.{f.name}").alias(f.name) 
                                               for f in field.dataType]).alias(field.name))
                elif isinstance(field.dataType, ArrayType):
                    # Handle array
                    if isinstance(field.dataType.elementType, StructType):
                        # Array of structs, recurse
                        temp_col = f"temp_{field.name}"
                        nested_df = align_array(df, curr_field.dataType, field.dataType, full_name)
                        output_fields.append(col(temp_col).alias(field.name))
                    else:
                        # Simple array, just cast
                        output_fields.append(col(full_name).cast(field.dataType).alias(field.name))
                elif isinstance(field.dataType, MapType):
                    # Handle map
                    if isinstance(field.dataType.valueType, StructType):
                        # Map with struct values, recurse
                        temp_col = f"temp_{field.name}"
                        nested_df = align_map(df, curr_field.dataType, field.dataType, full_name)
                        output_fields.append(col(temp_col).alias(field.name))
                    else:
                        # Simple map, just cast
                        output_fields.append(col(full_name).cast(field.dataType).alias(field.name))
                else:
                    # Simple type, cast to target type
                    output_fields.append(col(full_name).cast(field.dataType).alias(field.name))
        
        # Select all fields, preserving existing ones and adding new ones
        select_expr = []
        for field in target_schema:
            if any(f.name == field.name for f in curr_schema):
                select_expr.append(col(field.name))
            else:
                select_expr.append(lit(None).cast(field.dataType).alias(field.name))
                
        return df.select(*select_expr)
    
    def align_array(df, curr_type, target_type, col_name):
        """Handles arrays with nested structs."""
        if not isinstance(target_type.elementType, StructType):
            return df.withColumn("temp", col(col_name).cast(target_type)).drop(col_name)
        
        # Explode array to process struct elements
        exploded_df = df.withColumn("exploded", explode_outer(col(col_name)))
        
        # Align nested struct
        nested_df = align_struct(exploded_df, 
                               curr_type.elementType if isinstance(curr_type, ArrayType) else StructType(),
                               target_type.elementType,
                               "exploded.")
        
        # Collect back into array
        grouped_df = nested_df.groupBy(*[c for c in df.columns if c != col_name]).agg(
            collect_list(struct(*[col(f"exploded.{f.name}").alias(f.name) 
                                for f in target_type.elementType])).alias("temp")
        )
        
        return grouped_df.withColumn("temp", col("temp").cast(target_type))
    
    def align_map(df, curr_type, target_type, col_name):
        """Handles maps with nested structs."""
        if not isinstance(target_type.valueType, StructType):
            return df.withColumn("temp", col(col_name).cast(target_type)).drop(col_name)
        
        # Transform map to array of structs for processing
        transformed_df = df.withColumn("pairs", explode_outer(col(col_name)))
        
        # Align nested struct
        nested_df = align_struct(transformed_df, 
                               curr_type.valueType if isinstance(curr_type, MapType) else StructType(),
                               target_type.valueType,
                               "pairs.value.")
        
        # Reconstruct map
        grouped_df = nested_df.groupBy(*[c for c in df.columns if c != col_name]).agg(
            map_from_arrays(
                collect_list(col("pairs.key")),
                collect_list(struct(*[col(f"pairs.value.{f.name}").alias(f.name) 
                                   for f in target_type.valueType]))
            ).alias("temp")
        )
        
        return grouped_df.withColumn("temp", col("temp").cast(target_type))
    
    # Start alignment from root struct
    return align_struct(df, df.schema, target_schema)
