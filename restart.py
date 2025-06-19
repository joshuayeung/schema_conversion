from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct, lit, transform, array, when, coalesce, map_from_arrays, map_keys, map_values
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, MapType

# Initialize Spark session
spark = SparkSession.builder.appName("AddMissingFieldsComplexCorrected2").getOrCreate()

# Sample DataFrame with complex nested schema
data = [
    (1, ("Alice", [(25, ["F", "NB"], {"role1": (100,)})])),
    (2, ("Bob", [(30, [], {"role2": (200,)})])),
    (3, ("Cathy", [(28, ["F"], {})]))
]
schema = StructType([
    StructField("id", IntegerType()),
    StructField("info", StructType([
        StructField("name", StringType()),
        StructField("details", ArrayType(StructType([
            StructField("age", IntegerType()),
            StructField("tags", ArrayType(StringType())),
            StructField("roles", MapType(StringType(), StructType([
                StructField("code", IntegerType())
            ])))
        ])))
    ]))
])
df = spark.createDataFrame(data, schema)

# Target schema with additional fields and deeper nesting
target_schema = StructType([
    StructField("id", IntegerType()),
    StructField("info", StructType([
        StructField("name", StringType()),
        StructField("status", StringType()),  # New field
        StructField("details", ArrayType(StructType([
            StructField("age", IntegerType()),
            StructField("tags", ArrayType(StringType())),
            StructField("roles", MapType(StringType(), StructType([
                StructField("code", IntegerType()),
                StructField("desc", StringType()),  # New field in map struct
                StructField("extra", StructType([    # New nested struct
                    StructField("flag", StringType())
                ]))
            ]))),
            StructField("history", ArrayType(MapType(StringType(), StructType([  # New array of maps
                StructField("event", StringType())
            ]))))
        ]))),
        StructField("extra_nested", StructType([  # New deeply nested struct
            StructField("level1", StructType([
                StructField("level2", StructType([
                    StructField("level3", ArrayType(StructType([
                        StructField("value", IntegerType()),
                        StructField("meta", MapType(StringType(), StructType([
                            StructField("key", StringType())
                        ])))
                    ])))
                ]))
            ]))
        ]))
    ])),
    StructField("extra_field", StringType())  # New top-level field
])

# Show original DataFrame and schema
print("Original DataFrame:")
df.show(truncate=False)
df.printSchema()

def get_field_names(schema, prefix="", in_array=False, in_map=False):
    """Extract all field names from a schema, including nested structs, arrays, and maps."""
    fields = []
    for field in schema.fields:
        full_path = f"{prefix}.{field.name}" if prefix and not (in_array or in_map) else field.name
        fields.append(full_path)
        if isinstance(field.dataType, StructType):
            fields.extend(get_field_names(field.dataType, full_path, in_array, in_map))
        elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            fields.extend(get_field_names(field.dataType.elementType, full_path, in_array=True))
        elif isinstance(field.dataType, MapType) and isinstance(field.dataType.valueType, StructType):
            fields.extend(get_field_names(field.dataType.valueType, full_path, in_map=True))
    return fields

def build_struct_expr(target_schema, df_schema, prefix="", is_array_element=False, is_map_value=False, lambda_var=None):
    """
    Build a struct expression for a schema, using existing fields from df_schema
    and nulls for missing fields. Handles structs, arrays, and maps recursively.
    
    Args:
        target_schema: Target schema for the struct
        df_schema: Current DataFrame schema for the struct
        prefix: Column prefix (e.g., 'info.details')
        is_array_element: True if processing a struct within an array
        is_map_value: True if processing a struct within a map's value
        lambda_var: Lambda variable (e.g., 'x' for arrays, 'y' for maps)
    
    Returns:
        Struct expression with all fields from target_schema
    """
    from pyspark.sql.types import StructType, ArrayType, MapType

    # Get field names from DataFrame schema
    df_field_names = get_field_names(df_schema, in_array=is_array_element, in_map=is_map_value) if isinstance(df_schema, StructType) else []
    
    fields = []
    for field in target_schema.fields:
        full_path = f"{prefix}.{field.name}" if prefix and not (is_array_element or is_map_value) else field.name
        if isinstance(field.dataType, StructType):
            # Recurse into nested struct
            sub_df_schema = next((f.dataType for f in df_schema.fields if f.name == field.name), StructType([]))
            if (is_array_element or is_map_value) and lambda_var is not None:
                fields.append(build_struct_expr(field.dataType, sub_df_schema, "", is_array_element=is_array_element, is_map_value=is_map_value, lambda_var=lambda_var).alias(field.name))
            else:
                fields.append(build_struct_expr(field.dataType, sub_df_schema, full_path).alias(field.name))
        elif isinstance(field.dataType, ArrayType):
            # Handle arrays (of structs or other types)
            sub_df_schema = next((f.dataType.elementType for f in df_schema.fields if f.name == field.name), field.dataType.elementType)
            if isinstance(field.dataType.elementType, StructType):
                array_col = col(full_path) if full_path in df_field_names else array(lit(None).cast(field.dataType.elementType))
                fields.append(
                    when(
                        array_col.isNotNull(),
                        transform(
                            coalesce(array_col, array(lit(None).cast(field.dataType.elementType))),
                            lambda x: build_struct_expr(field.dataType.elementType, sub_df_schema, "", is_array_element=True, lambda_var=x)
                        )
                    ).otherwise(lit(None).cast(field.dataType)).alias(field.name)
                )
            else:
                # Non-struct arrays (e.g., ArrayType(StringType))
                if (is_array_element or is_map_value) and lambda_var is not None:
                    field_ref = getattr(lambda_var, field.name) if field.name in [f.name for f in df_schema.fields] else lit(None).cast(field.dataType)
                    fields.append(field_ref.alias(field.name))
                else:
                    fields.append(col(full_path).alias(field.name) if full_path in df_field_names else lit(None).cast(field.dataType).alias(field.name))
        elif isinstance(field.dataType, MapType):
            # Handle maps (with struct values or other types)
            if (is_array_element or is_map_value) and lambda_var is not None:
                # For maps within arrays or other maps, use lambda_var
                map_col = getattr(lambda_var, field.name) if field.name in [f.name for f in df_schema.fields] else lit(None).cast(field.dataType)
            else:
                map_col = col(full_path) if full_path in df_field_names else lit(None).cast(field.dataType)
            if isinstance(field.dataType.valueType, StructType):
                sub_df_schema = next((f.dataType.valueType for f in df_schema.fields if f.name == field.name), StructType([]))
                fields.append(
                    when(
                        map_col.isNotNull(),
                        map_from_arrays(
                            map_keys(coalesce(map_col, map_from_arrays(array(), array()))),
                            transform(
                                map_values(coalesce(map_col, map_from_arrays(array(), array()))),
                                lambda y: build_struct_expr(field.dataType.valueType, sub_df_schema, "", is_map_value=True, lambda_var=y)
                            )
                        )
                    ).otherwise(lit(None).cast(field.dataType)).alias(field.name)
                )
            else:
                # Non-struct map values
                fields.append(map_col.alias(field.name))
        else:
            # Simple types (e.g., IntegerType, StringType)
            if (is_array_element or is_map_value) and lambda_var is not None:
                field_ref = getattr(lambda_var, field.name) if field.name in [f.name for f in df_schema.fields] else lit(None).cast(field.dataType)
                fields.append(field_ref.alias(field.name))
            else:
                if full_path in df_field_names or field.name in [f.name for f in df_schema.fields]:
                    fields.append(col(full_path).alias(field.name))
                else:
                    fields.append(lit(None).cast(field.dataType).alias(field.name))
    return struct(*fields)

def align_dataframe_to_schema(df, target_schema):
    """
    Align DataFrame schema to target schema by adding missing fields recursively,
    including support for nested structs, arrays, and maps.
    
    Args:
        df: PySpark DataFrame
        target_schema: Target schema with additional fields
    
    Returns:
        Updated DataFrame with all fields from target_schema
    """
    from pyspark.sql.types import StructType, ArrayType, MapType

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
            elif isinstance(field.dataType, ArrayType):
                if isinstance(field.dataType.elementType, StructType):
                    df_updated = df_updated.withColumn(
                        field.name,
                        array(lit(None).cast(field.dataType.elementType))
                    )
                else:
                    df_updated = df_updated.withColumn(
                        field.name,
                        array(lit(None).cast(field.dataType.elementType))
                    )
            elif isinstance(field.dataType, MapType):
                df_updated = df_updated.withColumn(
                    field.name,
                    lit(None).cast(field.dataType)
                )
            else:
                df_updated = df_updated.withColumn(
                    field.name,
                    lit(None).cast(field.dataType)
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
                        lambda x: build_struct_expr(field.dataType.elementType, sub_df_schema, "", is_array_element=True, lambda_var=x)
                    )
                ).otherwise(lit(None).cast(field.dataType)).alias(field.name)
            )
        elif isinstance(field.dataType, MapType) and isinstance(field.dataType.valueType, StructType):
            df_sub_schema = next((f.dataType.valueType for f in df_updated.schema.fields if f.name == field.name), StructType([]))
            select_expr.append(
                when(
                    col(field.name).isNotNull(),
                    map_from_arrays(
                        map_keys(coalesce(col(field.name), map_from_arrays(array(), array()))),
                        transform(
                            map_values(coalesce(col(field.name), map_from_arrays(array(), array()))),
                            lambda y: build_struct_expr(field.dataType.valueType, sub_df_schema, "", is_map_value=True, lambda_var=y)
                        )
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
