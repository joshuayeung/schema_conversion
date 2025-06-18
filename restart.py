from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct, lit, transform, array, when, coalesce
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

# Initialize Spark session
spark = SparkSession.builder.appName("AddMissingFieldsWithArraysFixed").getOrCreate()

# Sample DataFrame with nested structs and arrays
data = [
    (1, ("Alice", [(25, "F")]), (["x", "y"])),
    (2, ("Bob", [(30, "M")]), ([])),
    (3, ("Cathy", [(28, "F")]), (["z"]))
]
schema = StructType([
    StructField("id", IntegerType()),
    StructField("info", StructType([
        StructField("name", StringType()),
        StructField("details", ArrayType(StructType([
            StructField("age", IntegerType()),
            StructField("gender", StringType())
        ])))
    ])),
    StructField("tags", ArrayType(StringType()))
])
df = spark.createDataFrame(data, schema)

# Target schema with additional fields
target_schema = StructType([
    StructField("id", IntegerType()),
    StructField("info", StructType([
        StructField("name", StringType()),
        StructField("status", StringType()),  # New field
        StructField("details", ArrayType(StructType([
            StructField("age", IntegerType()),
            StructField("gender", StringType()),
            StructField("status", StringType()),  # New field in array struct
            StructField("extra", StructType([     # New nested struct in array
                StructField("code", IntegerType()),
                StructField("flag", StringType())
            ]))
        ]))),
        StructField("extra_array", ArrayType(StructType([  # New array of structs
            StructField("id", IntegerType()),
            StructField("desc", StringType())
        ])))
    ])),
    StructField("tags", ArrayType(StringType())),
    StructField("extra_field", StringType())  # New top-level field
])

# Show original DataFrame and schema
print("Original DataFrame:")
df.show(truncate=False)
df.printSchema()

def get_field_names(schema, prefix=""):
    """Extract all field names from a schema, including nested structs and arrays."""
    fields = []
    for field in schema.fields:
        full_path = f"{prefix}.{field.name}" if prefix else field.name
        fields.append(full_path)
        if isinstance(field.dataType, StructType):
            fields.extend(get_field_names(field.dataType, full_path))
        elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            fields.extend(get_field_names(field.dataType.elementType, full_path))
    return fields

def build_struct_expr(target_schema, df_schema, prefix="", is_array_element=False):
    """
    Build a struct expression for a schema, using existing fields from df_schema
    and nulls for missing fields. Handles structs within arrays.
    
    Args:
        target_schema: Target schema for the struct
        df_schema: Current DataFrame schema for the struct
        prefix: Column prefix (e.g., 'info.details')
        is_array_element: True if processing a struct within an array
    
    Returns:
        Struct expression with all fields from target_schema
    """
    from pyspark.sql.types import StructType, ArrayType

    # Get field names from DataFrame schema
    df_field_names = get_field_names(df_schema) if isinstance(df_schema, StructType) else []
    
    fields = []
    for field in target_schema.fields:
        full_path = f"{prefix}.{field.name}" if prefix and not is_array_element else field.name
        if isinstance(field.dataType, StructType):
            # Recurse into nested struct
            sub_df_schema = next((f.dataType for f in df_schema.fields if f.name == field.name), StructType([]))
            fields.append(build_struct_expr(field.dataType, sub_df_schema, full_path).alias(field.name))
        elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            # Handle array of structs
            sub_df_schema = next((f.dataType.elementType for f in df_schema.fields if f.name == field.name), StructType([]))
            array_col = col(full_path) if full_path in df_field_names else array(lit(None).cast(field.dataType.elementType))
            fields.append(
                when(
                    array_col.isNotNull(),
                    transform(
                        coalesce(array_col, array(lit(None).cast(field.dataType.elementType))),
                        lambda x: build_struct_expr(field.dataType.elementType, sub_df_schema, "", is_array_element=True)
                    )
                ).otherwise(lit(None).cast(field.dataType)).alias(field.name)
            )
        else:
            # Use existing field if available, otherwise use null
            if full_path in df_field_names or field.name in [f.name for f in df_schema.fields]:
                fields.append(col(full_path).alias(field.name))
            else:
                fields.append(lit(None).cast(field.dataType).alias(field.name))
    return struct(*fields)

def align_dataframe_to_schema(df, target_schema):
    """
    Align DataFrame schema to target schema by adding missing fields recursively,
    including support for arrays of structs.
    
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
            elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
                df_updated = df_updated.withColumn(
                    field.name,
                    array(lit(None).cast(field.dataType.elementType))
                )
            else:
                df_updated = df_updated.withColumn(
                    field.name, lit(None).cast(field.dataType)
                )

    # Build select expression to match target schema
    select_expr = []
    for field in target_schema.fields:
        if isinstance(field.dataType, StructType):
            df_sub_schema = next((f.dataType for f in df_updated.schema.fields if f.name == field.name), StructType([]))
            select_expr.append(build_struct_expr(field.dataType, df_sub_schema, field.name).alias(field.name))
        elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            df_sub_schema = next((f.dataType.elementType for f in df_updated.schema.fields if f.name == field.name), StructType([]))
            select_expr.append(
                when(
                    col(field.name).isNotNull(),
                    transform(
                        coalesce(col(field.name), array(lit(None).cast(field.dataType.elementType))),
                        lambda x: build_struct_expr(field.dataType.elementType, df_sub_schema, "", is_array_element=True)
                    )
                ).otherwise(lit(None).cast(field.dataType)).alias(field.name)
            )
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
