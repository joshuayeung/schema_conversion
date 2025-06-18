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

def add_missing_fields(df, df_schema, target_schema, column_prefix=""):
    """
    Recursively add missing fields from target_schema to df's nested structs.
    
    Args:
        df: PySpark DataFrame
        df_schema: Current schema of the DataFrame
        target_schema: Target schema with additional fields
        column_prefix: Prefix for nested column references (e.g., 'info.details')
    
    Returns:
        Updated DataFrame with missing fields added
    """
    from pyspark.sql.types import StructType

    # If schemas are not StructType, return df unchanged (base case for non-struct types)
    if not isinstance(df_schema, StructType) or not isinstance(target_schema, StructType):
        return df

    # Iterate through target schema fields
    for field in target_schema.fields:
        # Full column path for the field
        current_path = f"{column_prefix}.{field.name}" if column_prefix else field.name
        
        # Check if field exists in current DataFrame schema
        df_field = None
        for f in df_schema.fields:
            if f.name == field.name:
                df_field = f
                break

        if df_field is None:
            # Field is missing; add it with null or default value
            if isinstance(field.dataType, StructType):
                # For missing structs, create a null struct with the target schema
                df = df.withColumn(
                    current_path,
                    struct(*[lit(None).cast(subfield.dataType).alias(subfield.name) 
                             for subfield in field.dataType.fields])
                )
            else:
                # For non-struct types, add null or default value
                df = df.withColumn(current_path, lit(None).cast(field.dataType))
        elif isinstance(field.dataType, StructType):
            # Field exists and is a struct; recurse into its fields
            df = add_missing_fields(
                df,
                df_field.dataType,
                field.dataType,
                current_path
            )

    return df

def align_dataframe_to_schema(df, target_schema):
    """
    Align DataFrame schema to target schema by adding missing fields.
    
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
        elif isinstance(field.dataType, StructType):
            # Recurse into top-level struct fields
            df_updated = add_missing_fields(
                df_updated,
                df.schema[field.name].dataType,
                field.dataType,
                field.name
            )
    
    # Reconstruct the DataFrame with the target schema order
    def build_struct_expr(schema, prefix=""):
        """Helper to build struct expressions for nested fields"""
        if not isinstance(schema, StructType):
            return col(f"{prefix}") if prefix else None
        fields = []
        for field in schema.fields:
            full_path = f"{prefix}.{field.name}" if prefix else field.name
            if isinstance(field.dataType, StructType):
                fields.append(build_struct_expr(field.dataType, full_path).alias(field.name))
            else:
                fields.append(col(full_path).alias(field.name))
        return struct(*fields)

    # Select columns in target schema order
    select_expr = []
    for field in target_schema.fields:
        if isinstance(field.dataType, StructType):
            select_expr.append(build_struct_expr(field.dataType, field.name).alias(field.name))
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
