from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct, lit, transform, array, when, coalesce, map_from_arrays, map_keys, map_values
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, MapType, BooleanType, FloatType, DoubleType, LongType

# Initialize Spark session
spark = SparkSession.builder.appName("AddMissingFieldsComplexCorrected35").getOrCreate()

# Sample DataFrame with complex nested schema (jumiokyx not present)
data = [
    (1, {"jumio": None}),  # vendorResult.jumio is NULL
    (2, {"jumio": {"verification": {"someOtherField": "value"}}}),
    (3, {"jumio": {"verification": {"someOtherField": "value2"}}})
]
schema = StructType([
    StructField("id", IntegerType()),
    StructField("vendorResult", StructType([
        StructField("jumio", StructType([
            StructField("verification", StructType([
                StructField("someOtherField", StringType())
            ]))
        ]))
    ]))
])
df = spark.createDataFrame(data, schema)

# Target schema with jumiokyx added, kyxData non-nullable
target_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("vendorResult", StructType([
        StructField("jumio", StructType([
            StructField("verification", StructType([
                StructField("someOtherField", StringType(), True)
            ]), True)
        ]), True),
        StructField("jumiokyx", StructType([
            StructField("kyxData", StructType([
                StructField("checks", ArrayType(StructType([
                    StructField("checkId", StringType(), True),
                    StructField("status", StringType(), False)
                ]), True), True)
            ]), False)
        ]), True)
    ]), True),
    StructField("extra_field", StringType(), True)
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

def print_schema_structure(schema, indent=0):
    """Print the schema structure with indentation for debugging."""
    for field in schema.fields:
        print("  " * indent + f"Field: {field.name}, Type: {field.dataType.simpleString()}, Nullable: {field.nullable}")
        if isinstance(field.dataType, StructType):
            print_schema_structure(field.dataType, indent + 1)
        elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            print("  " * (indent + 1) + "Array element:")
            print_schema_structure(field.dataType.elementType, indent + 2)

def validate_schema(schema, prefix=""):
    """Validate schema to ensure no duplicate or incorrect fields."""
    from pyspark.sql.types import StructType
    for field in schema.fields:
        full_path = f"{prefix}.{field.name}" if prefix else field.name
        if isinstance(field.dataType, StructType):
            subfields = [f.name for f in field.dataType.fields]
            if field.name in subfields:
                print(f"Validation error: Field {field.name} appears in its own subfields {subfields} at {full_path}")
                raise ValueError(f"Invalid schema: {field.name} cannot be a subfield of itself at {full_path}")
            validate_schema(field.dataType, full_path)

def generate_default_struct(field_type):
    """
    Generate a default struct expression for a given schema, handling non-nullable fields recursively.
    
    Args:
        field_type: DataType of the field (e.g., StructType, ArrayType, StringType)
    
    Returns:
        PySpark Column expression with default values
    """
    from pyspark.sql.types import StructType, ArrayType, MapType, StringType, IntegerType, BooleanType, FloatType, DoubleType, LongType

    if isinstance(field_type, StructType):
        default_fields = []
        for subfield in field_type.fields:
            if not subfield.nullable:
                # Non-nullable field: provide default value
                default_value = generate_default_struct(subfield.dataType)
                default_fields.append(default_value.alias(subfield.name))
                print(f"Assigned default value for non-nullable field {subfield.name}: {default_value}")
            else:
                # Nullable field: use NULL
                default_fields.append(lit(None).cast(subfield.dataType).alias(subfield.name))
        return struct(*default_fields)
    elif isinstance(field_type, ArrayType):
        if not field_type.containsNull:
            # Non-nullable array: provide empty array with default element
            element_default = generate_default_struct(field_type.elementType)
            return array(element_default).cast(field_type)
        else:
            # Nullable array: use NULL
            return lit(None).cast(field_type)
    elif isinstance(field_type, MapType):
        # Maps are typically nullable; use NULL
        return lit(None).cast(field_type)
    else:
        # Simple types: provide default values for non-nullable fields
        if isinstance(field_type, StringType):
            return lit("")  # Default for non-nullable string
        elif isinstance(field_type, IntegerType) or isinstance(field_type, LongType):
            return lit(0)  # Default for non-nullable integer/long
        elif isinstance(field_type, BooleanType):
            return lit(False)  # Default for non-nullable boolean
        elif isinstance(field_type, FloatType) or isinstance(field_type, DoubleType):
            return lit(0.0)  # Default for non-nullable float/double
        else:
            return lit(None).cast(field_type)  # Nullable or unknown types

def build_struct_expr(target_schema, df_schema, prefix="", is_array_element=False, is_map_value=False, lambda_var=None):
    """
    Build a struct expression for a schema, using existing fields from df_schema
    and default values for missing fields. Handles structs, arrays, and maps recursively.
    
    Args:
        target_schema: Target schema for the struct
        df_schema: Current DataFrame schema for the struct
        prefix: Column prefix (e.g., 'vendorResult')
        is_array_element: True if processing a struct within an array
        is_map_value: True if processing a struct within a map's value
        lambda_var: Lambda variable (e.g., 'x' for arrays, 'y' for maps)
    
    Returns:
        Struct expression with all fields from target_schema
    """
    from pyspark.sql.types import StructType, ArrayType, MapType

    # Get field names from DataFrame schema
    df_field_names = get_field_names(df_schema) if isinstance(df_schema, StructType) else []
    
    # Log the current schema and prefix
    print(f"Building struct expr for prefix: {prefix}, is_array_element: {is_array_element}, is_map_value: {is_map_value}")
    print(f"Target schema fields: {[f.name for f in target_schema.fields]}")
    print(f"DataFrame schema fields: {[f.name for f in df_schema.fields] if isinstance(df_schema, StructType) else []}")
    
    fields = []
    for field in target_schema.fields:
        full_path = f"{prefix}.{field.name}" if prefix and not (is_array_element or is_map_value) else field.name
        print(f"Processing field: {field.name}, full_path: {full_path}")
        
        if isinstance(field.dataType, StructType):
            # Recurse into nested struct
            sub_df_schema = next((f.dataType for f in df_schema.fields if f.name == field.name), StructType([]))
            print(f"Subschema for {field.name}: {[f.name for f in sub_df_schema.fields] if isinstance(sub_df_schema, StructType) else []}")
            if (is_array_element or is_map_value) and lambda_var is not None:
                field_ref = getattr(lambda_var, field.name)
                if field.nullable:
                    field_expr = when(
                        field_ref.isNotNull(),
                        build_struct_expr(field.dataType, sub_df_schema, "", is_array_element=is_array_element, is_map_value=is_map_value, lambda_var=field_ref)
                    ).otherwise(generate_default_struct(field.dataType) if not field.nullable else struct(*[lit(None).cast(subfield.dataType).alias(subfield.name) for subfield in field.dataType.fields]))
                else:
                    # Non-nullable struct: provide default values
                    field_expr = build_struct_expr(field.dataType, sub_df_schema, "", is_array_element=is_array_element, is_map_value=is_map_value, lambda_var=field_ref)
            else:
                if field.nullable:
                    field_expr = when(
                        col(full_path).isNotNull(),
                        build_struct_expr(field.dataType, sub_df_schema, full_path, is_array_element, is_map_value, lambda_var)
                    ).otherwise(generate_default_struct(field.dataType) if not field.nullable else struct(*[lit(None).cast(subfield.dataType).alias(subfield.name) for subfield in field.dataType.fields]))
                else:
                    # Non-nullable struct: provide default values
                    field_expr = build_struct_expr(field.dataType, sub_df_schema, full_path, is_array_element, is_map_value, lambda_var)
            fields.append(field_expr.alias(field.name))
            print(f"Generated expr for {field.name}: {field_expr}")
        elif isinstance(field.dataType, ArrayType):
            # Handle arrays (of structs or other types)
            sub_df_schema = next((f.dataType.elementType for f in df_schema.fields if f.name == field.name), field.dataType.elementType)
            print(f"Subschema for {field.name}: {[f.name for f in sub_df_schema.fields] if isinstance(sub_df_schema, StructType) else []}")
            if isinstance(field.dataType.elementType, StructType):
                if (is_array_element or is_map_value) and lambda_var is not None:
                    array_col = getattr(lambda_var, field.name)
                else:
                    array_col = col(full_path)
                field_expr = when(
                    array_col.isNotNull(),
                    transform(
                        array_col,
                        lambda x: build_struct_expr(field.dataType.elementType, sub_df_schema, "", is_array_element=True, lambda_var=x)
                    )
                ).otherwise(lit(None).cast(field.dataType)).alias(field.name)
                fields.append(field_expr)
                print(f"Generated expr for {field.name}: {field_expr}")
            else:
                if (is_array_element or is_map_value) and lambda_var is not None:
                    field_ref = getattr(lambda_var, field.name)
                    fields.append(field_ref.alias(field.name))
                    print(f"Generated expr for {field.name}: {field_ref}")
                else:
                    field_ref = col(full_path)
                    fields.append(field_ref.alias(field.name))
                    print(f"Generated expr for {field.name}: {field_ref}")
        elif isinstance(field.dataType, MapType):
            # Handle maps (with struct values or other types)
            if (is_array_element or is_map_value) and lambda_var is not None:
                map_col = getattr(lambda_var, field.name)
            else:
                map_col = col(full_path)
            if isinstance(field.dataType.valueType, StructType):
                sub_df_schema = next((f.dataType.valueType for f in df_schema.fields if f.name == field.name), field.dataType.valueType)
                field_expr = when(
                    map_col.isNotNull(),
                    map_from_arrays(
                        map_keys(coalesce(map_col, map_from_arrays(array(), array()))),
                        transform(
                            map_values(coalesce(map_col, map_from_arrays(array(), array()))),
                            lambda y: build_struct_expr(field.dataType.valueType, sub_df_schema, "", is_map_value=True, lambda_var=y)
                        )
                    )
                ).otherwise(lit(None).cast(field.dataType)).alias(field.name)
                fields.append(field_expr)
                print(f"Generated expr for {field.name}: {field_expr}")
            else:
                fields.append(map_col.alias(field.name))
                print(f"Generated expr for {field.name}: {map_col}")
        else:
            # Simple types (e.g., IntegerType, StringType)
            if (is_array_element or is_map_value) and lambda_var is not None:
                field_ref = getattr(lambda_var, field.name)
                fields.append(field_ref.alias(field.name))
                print(f"Generated expr for {field.name}: {field_ref}")
            else:
                field_ref = col(full_path)
                fields.append(field_ref.alias(field.name))
                print(f"Generated expr for {field.name}: {field_ref}")
    return struct(*fields)

def clean_null_structs(df, schema, prefix=""):
    """
    Recursively set nullable struct fields to NULL if all their subfields are NULL.
    
    Args:
        df: PySpark DataFrame
        schema: Schema to process
        prefix: Current column prefix (e.g., 'vendorResult')
    
    Returns:
        Updated DataFrame with cleaned structs
    """
    from pyspark.sql.types import StructType

    df_updated = df
    print(f"Processing clean_null_structs at prefix: {prefix}")
    print("Schema structure:")
    print_schema_structure(schema)

    for field in schema.fields:
        full_path = f"{prefix}.{field.name}" if prefix else field.name
        if not isinstance(field.dataType, StructType):
            print(f"Skipping non-struct field: {full_path}")
            continue
        print(f"Processing struct field: {full_path}, nullable: {field.nullable}")
        print(f"Subfields for {full_path}: {[subfield.name for subfield in field.dataType.fields]}")
        if field.nullable:
            # Generate condition: all subfields are NULL
            subfields = field.dataType.fields
            if subfields:  # Only process structs with subfields
                null_conditions = []
                for subfield in subfields:
                    subfield_path = f"{full_path}.{subfield.name}"
                    print(f"Checking null condition for: {subfield_path}")
                    null_conditions.append(col(subfield_path).isNull())
                if null_conditions:
                    all_null_condition = null_conditions[0]
                    for condition in null_conditions[1:]:
                        all_null_condition = all_null_condition & condition
                    print(f"Applying cleanup for {full_path} with condition: {all_null_condition}")
                    df_updated = df_updated.withColumn(
                        full_path,
                        when(all_null_condition, lit(None).cast(field.dataType)).otherwise(col(full_path))
                    )
                    print(f"Applied cleanup for {full_path}")
                else:
                    print(f"No subfields to check for {full_path}, skipping cleanup")
            else:
                print(f"No subfields for {full_path}, skipping cleanup")
        else:
            print(f"Skipping non-nullable struct: {full_path}")
        # Recurse into nested structs (nullable or non-nullable)
        df_updated = clean_null_structs(df_updated, field.dataType, full_path)

    return df_updated

def add_missing_nested_fields(df, target_schema, prefix="", df_schema=None):
    """
    Recursively add missing fields to the DataFrame at all levels of the schema,
    setting nullable structs to NULL and non-nullable structs to default values.
    
    Args:
        df: PySpark DataFrame
        target_schema: Target schema
        prefix: Current column prefix (e.g., 'vendorResult')
        df_schema: Current DataFrame schema (defaults to df.schema)
    
    Returns:
        Tuple of (updated DataFrame, updated schema)
    """
    from pyspark.sql.types import StructType, ArrayType, MapType

    if df_schema is None:
        df_schema = df.schema

    df_updated = df
    updated_schema_fields = list(df_schema.fields) if isinstance(df_schema, StructType) else []

    print(f"Processing schema at prefix: {prefix}, target fields: {[f.name for f in target_schema.fields]}")
    print(f"Current schema fields: {[f.name for f in updated_schema_fields]}")

    for field in target_schema.fields:
        full_path = f"{prefix}.{field.name}" if prefix else field.name
        field_name = field.name
        df_field_names = [f.name for f in updated_schema_fields]

        print(f"Checking field: {field_name}, full_path: {full_path}, in df_field_names: {field_name in df_field_names}")

        if field_name not in df_field_names:
            # Add missing field
            print(f"Adding missing field: {full_path}")
            if isinstance(field.dataType, StructType):
                if prefix:
                    # Add struct (e.g., jumiokyx) within parent struct (e.g., vendorResult)
                    parent_field = prefix.split('.')[-1]
                    # Find parent field in updated_schema_fields
                    parent_schema = next((f for f in updated_schema_fields if f.name == parent_field), None)
                    if parent_schema and isinstance(parent_schema.dataType, StructType):
                        parent_fields = parent_schema.dataType.fields
                        new_parent_fields = list(parent_fields) + [StructField(field_name, field.dataType, field.nullable)]
                        new_parent_schema = StructType(new_parent_fields)
                        struct_fields = []
                        for f in parent_fields:
                            struct_fields.append(col(f"{prefix}.{f.name}").alias(f.name))
                        struct_fields.append(generate_default_struct(field.dataType).alias(field_name))
                        print(f"Initialized {full_path} with default struct: {struct_fields[-1]}")
                        df_updated = df_updated.withColumn(
                            parent_field,
                            when(
                                col(parent_field).isNotNull(),
                                struct(*struct_fields)
                            ).otherwise(lit(None).cast(new_parent_schema))
                        )
                        print(f"Updated {parent_field} with {field_name}")
                        # Update the schema using updated_schema_fields
                        updated_schema_fields = [
                            StructField(f.name, new_parent_schema if f.name == parent_field else f.dataType, f.nullable)
                            for f in updated_schema_fields
                        ]
                    else:
                        print(f"Error: Parent field {parent_field} not found or not a struct at prefix {prefix}. Skipping field addition.")
                        continue
                elif field.nullable:
                    # Top-level nullable structs
                    df_updated = df_updated.withColumn(
                        field_name,
                        lit(None).cast(field.dataType)
                    )
                    print(f"Initialized {full_path} with NULL")
                    updated_schema_fields.append(StructField(field_name, field.dataType, field.nullable))
                else:
                    # Non-nullable top-level structs
                    default_struct = generate_default_struct(field.dataType)
                    df_updated = df_updated.withColumn(
                        field_name,
                        default_struct
                    )
                    print(f"Initialized {full_path} with default struct: {default_struct}")
                    updated_schema_fields.append(StructField(field_name, field.dataType, field.nullable))
            elif isinstance(field.dataType, ArrayType):
                df_updated = df_updated.withColumn(
                    full_path,
                    lit(None).cast(field.dataType)
                )
                print(f"Initialized {full_path} with [NULL]")
                updated_schema_fields.append(StructField(field_name, field.dataType, field.nullable))
            elif isinstance(field.dataType, MapType):
                df_updated = df_updated.withColumn(
                    full_path,
                    lit(None).cast(field.dataType)
                )
                print(f"Initialized {full_path} with NULL")
                updated_schema_fields.append(StructField(field_name, field.dataType, field.nullable))
            else:
                df_updated = df_updated.withColumn(
                    full_path,
                    lit(None).cast(field.dataType)
                )
                print(f"Initialized {full_path} with NULL")
                updated_schema_fields.append(StructField(field_name, field.dataType, field.nullable))
        elif isinstance(field.dataType, StructType):
            # Recurse into existing nested structs
            sub_df_schema = next((f.dataType for f in df_schema.fields if f.name == field_name), StructType([]))
            sub_df_updated, sub_df_schema = add_missing_nested_fields(df_updated, field.dataType, full_path, sub_df_schema)
            df_updated = sub_df_updated
            updated_schema_fields = [
                StructField(f.name, sub_df_schema if f.name == field_name else f.dataType, f.nullable)
                for f in updated_schema_fields
            ]
        
        # Log intermediate schema
        print(f"Schema after processing {full_path}:")
        print_schema_structure(StructType(updated_schema_fields))

    updated_schema = StructType(updated_schema_fields)
    print("Validating updated schema:")
    validate_schema(updated_schema)
    return df_updated, updated_schema

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

    # Add missing fields at all levels
    df_updated, updated_schema = add_missing_nested_fields(df, target_schema)
    print("Schema after add_missing_nested_fields:")
    df_updated.printSchema()
    print("Updated schema structure:")
    print_schema_structure(updated_schema)

    # Clean up nullable structs where all subfields are NULL
    df_updated = clean_null_structs(df_updated, updated_schema)
    print("Schema after clean_null_structs:")
    df_updated.printSchema()

    # Build select expression to match target schema
    select_expr = []
    for field in target_schema.fields:
        full_path = field.name
        if isinstance(field.dataType, StructType):
            df_sub_schema = next((f.dataType for f in updated_schema.fields if f.name == field.name), StructType([]))
            expr = build_struct_expr(field.dataType, df_sub_schema, field.name).alias(field.name)
            select_expr.append(expr)
        elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
            df_sub_schema = next((f.dataType.elementType for f in updated_schema.fields if f.name == field.name), field.dataType.elementType)
            expr = when(
                col(field.name).isNotNull(),
                transform(
                    col(field.name),
                    lambda x: build_struct_expr(field.dataType.elementType, df_sub_schema, "", is_array_element=True, lambda_var=x)
                )
            ).otherwise(lit(None).cast(field.dataType)).alias(field.name)
            select_expr.append(expr)
        elif isinstance(field.dataType, MapType) and isinstance(field.dataType.valueType, StructType):
            df_sub_schema = next((f.dataType.valueType for f in updated_schema.fields if f.name == field.name), field.dataType.valueType)
            expr = when(
                col(field.name).isNotNull(),
                map_from_arrays(
                    map_keys(coalesce(col(field.name), map_from_arrays(array(), array()))),
                    transform(
                        map_values(coalesce(col(field.name), map_from_arrays(array(), array()))),
                        lambda y: build_struct_expr(field.dataType.valueType, df_sub_schema, "", is_map_value=True, lambda_var=y)
                    )
                )
            ).otherwise(lit(None).cast(field.dataType)).alias(field.name)
            select_expr.append(expr)
        else:
            expr = col(field.name)
            select_expr.append(expr)
    
    # Log the select_expr
    print("Generated select_expr:")
    for expr in select_expr:
        print(expr)
    
    df_final = df_updated.select(*select_expr)
    
    # Enforce target schema
    df_final_enforced = df_final.sparkSession.createDataFrame(df_final.rdd, target_schema)
    
    return df_final_enforced

# Apply the function to align DataFrame to target schema
df_updated = align_dataframe_to_schema(df, target_schema)

# Show updated DataFrame and schema
print("Updated DataFrame:")
df_updated.show(truncate=False)
df_updated.printSchema()

# Stop Spark session
spark.stop()
