from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Initialize Spark session
spark = SparkSession.builder.appName("AddMissingFieldsRecursively").getOrCreate()

# Sample DataFrame with nested structs
data = [
    (1, ("Alice", (25,))),
    (2, ("Bob", (30,))),
    (3, ("Cathy", (28,)))
]
schema = StructType([
    StructField("id", IntegerType()),
    StructField("info", StructType([
        StructField("name", StringType()),
        StructField("details", StructType([
            StructField("age", IntegerType())
        ]))
    ]))
])
df = spark.createDataFrame(data, schema)

# Target schema with additional fields
target_schema = StructType([
    StructField("id", IntegerType()),
    StructField("info", StructType([
        StructField("name", StringType()),
        StructField("status", StringType()),  # New field
        StructField("details", StructType([
            StructField("age", IntegerType()),
            StructField("gender", StringType()),  # New field
            StructField("extra", StructType([     # New nested struct
                StructField("code", IntegerType()),
                StructField("flag", StringType())
            ]))
        ]))
    ]))
])

# Show original DataFrame and schema
print("Original DataFrame:")
df.show(truncate=False)
df.printSchema()

def get_field_names(schema):
    """Extract all field names from a schema, including nested structs."""
    fields = []
    for field in schema.fields:
        fields.append(field.name)
        if isinstance(field.dataType, StructType):
            nested_fields = get_field_names(field.dataType)
            fields.extend(f"{field.name}.{subfield}" for subfield in nested_fields)
    return fields

def build_struct_expr(target_schema, df_schema, prefix=""):
    """
    Build a struct expression for a schema, using existing fields from df_schema
    and nulls for missing fields.
    
    Args:
        target_schema: Target schema for the struct
        df_schema: Current DataFrame schema for the struct
        prefix: Column prefix (e.g., 'info.details')
    
    Returns:
        Struct expression with all fields from target_schema
    """
    from pyspark.sql.types import StructType

    df_field_names = get_field_names(df_schema) if isinstance(df_schema, StructType) else []
    fields = []
    for field in target_schema.fields:
        full_path = f"{prefix}.{field.name}" if prefix else field.name
        if isinstance(field.dataType, StructType):
            # Recurse into nested struct
            sub_df_schema = next((f.dataType for f in df_schema.fields if f.name == field.name), StructType([]))
            fields.append(build_struct_expr(field.dataType, sub_df_schema, full_path).alias(field.name))
        else:
            # Use existing field if available, otherwise use null
            if full_path in df_field_names or field.name in [f.name for f in df_schema.fields]:
                fields.append(col(full_path).alias(field.name))
            else:
                fields.append(lit(None).cast(field.dataType).alias(field.name))
    return struct(*fields)

def align_dataframe_to_schema(df, target_schema):
    """
    Align DataFrame schema to target schema by adding missing fields recursively.
    
    Args:
        df: PySpark DataFrame
        target_schema: Target schema with additional fields
    
    Returns:
        Updated DataFrame with all fields from target_schema
    """
    # Process top-level fields
    df_updated = df
    for field in target_schema.fields:
        if field.name not in [f.name for f in df.schema.fields]:
            # Add missing top-level field
            if isinstance(field.dataType, StructType):
                df_updated = df_updated.withColumn(
                    field.name,
                    struct(*[lit(None).cast(subfield.dataType).alias(subfield.name) 
                             for subfield in field.dataType.fields])
                )
            else:
                df_updated = df_updated.withColumn(
                    field.name, lit(None).cast(field.dataType)
                )

    # Build select expression to match target schema
    select_expr = []
    for field in target_schema.fields:
        if isinstance(field.dataType, StructType):
            # Get the sub-schema for the field from the DataFrame
            df_sub_schema = next((f.dataType for f in df_updated.schema.fields if f.name == field.name), StructType([]))
            select_expr.append(build_struct_expr(field.dataType, df_sub_schema, field.name).alias(field.name))
        else:
            select_expr.append(col(field.name))
    
    df_final = df_updated.select(*select_expr)
    return df_final

# Apply the function to align DataFrame to target schema
df_updated = align_dataframe_to_schema(df, target_schema)

# Show updated DataFrame and schema
print("Updated DataFrame:")
df_updated.show(truncate=False)
df_updated.printSchema()

# Stop Spark session
spark.stop()
