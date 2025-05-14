from pyspark.sql import DataFrame
from pyspark.sql.functions import col, struct, transform, coalesce
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, StringType, IntegerType


def merge_with_iceberg_schema(df: DataFrame, iceberg_schema: StructType) -> DataFrame:
    """
    Merge a DataFrame's schema with an Iceberg table's schema, treating the Iceberg schema
    as the gold standard. For case-insensitive duplicates, use the Iceberg schema's field.
    Include non-duplicate fields from the DataFrame.

    Args:
        df: Input PySpark DataFrame
        iceberg_schema: Iceberg table schema (StructType)
    Returns:
        DataFrame with merged schema
    """

    def normalize_schema(schema: StructType):
        """Create a case-insensitive mapping of field names to their original fields."""
        normalized = {}
        for field in schema.fields:
            field_lower = field.name.lower()
            normalized[field_lower] = field
        return normalized

    def build_merged_schema(df_schema: StructType, ice_schema: StructType):
        """Recursively build the merged schema."""
        ice_normalized = normalize_schema(ice_schema)
        df_normalized = normalize_schema(df_schema)

        new_fields = []
        # First, include all Iceberg fields
        for ice_field in ice_schema.fields:
            field_lower = ice_field.name.lower()
            new_field = ice_field

            if field_lower in df_normalized:
                df_field = df_normalized[field_lower]
                # For nested types, recurse
                if isinstance(ice_field.dataType, StructType) and isinstance(df_field.dataType, StructType):
                    nested_schema = build_merged_schema(df_field.dataType, ice_field.dataType)
                    new_field = StructField(ice_field.name, nested_schema, ice_field.nullable)
                elif (isinstance(ice_field.dataType, ArrayType) and
                      isinstance(ice_field.dataType.elementType, StructType) and
                      isinstance(df_field.dataType, ArrayType) and
                      isinstance(df_field.dataType.elementType, StructType)):
                    nested_schema = build_merged_schema(df_field.dataType.elementType,
                                                        ice_field.dataType.elementType)
                    new_field = StructField(ice_field.name, ArrayType(nested_schema), ice_field.nullable)
                elif (isinstance(ice_field.dataType, MapType) and
                      isinstance(ice_field.dataType.valueType, StructType) and
                      isinstance(df_field.dataType, MapType) and
                      isinstance(df_field.dataType.valueType, StructType)):
                    nested_schema = build_merged_schema(df_field.dataType.valueType,
                                                        ice_field.dataType.valueType)
                    new_field = StructField(ice_field.name,
                                            MapType(ice_field.dataType.keyType, nested_schema),
                                            ice_field.nullable)

            new_fields.append(new_field)

        # Add non-duplicate DataFrame fields
        for df_field in df_schema.fields:
            if df_field.name.lower() not in ice_normalized:
                new_fields.append(df_field)

        return StructType(new_fields)

    def build_select_expr(merged_schema: StructType, df_schema: StructType, ice_schema: StructType, prefix: str = ""):
        """Build select expressions to map DataFrame columns to merged schema."""
        ice_normalized = normalize_schema(ice_schema)
        df_normalized = normalize_schema(df_schema)

        select_exprs = []
        for field in merged_schema.fields:
            field_lower = field.name.lower()
            full_name = f"{prefix}{field.name}"

            if field_lower in ice_normalized:
                # Field from Iceberg schema
                ice_field = ice_normalized[field_lower]
                ice_full_name = f"{prefix}{ice_field.name}"

                if isinstance(field.dataType, StructType):
                    nested_schema = field.dataType
                    nested_df_schema = df_normalized.get(field_lower, StructField("", StructType([]))).dataType
                    nested_ice_schema = ice_field.dataType
                    nested_exprs = build_select_expr(nested_schema, nested_df_schema, nested_ice_schema,
                                                     f"{ice_full_name}.")
                    select_exprs.append(struct(nested_exprs).alias(field.name))
                elif (isinstance(field.dataType, ArrayType) and
                      isinstance(field.dataType.elementType, StructType)):
                    nested_schema = field.dataType.elementType
                    nested_df_schema = df_normalized.get(field_lower, StructField("", ArrayType(
                        StructType([])))).dataType.elementType
                    nested_ice_schema = ice_field.dataType.elementType
                    nested_exprs = build_select_expr(nested_schema, nested_df_schema, nested_ice_schema, "")
                    select_exprs.append(transform(col(ice_full_name), lambda x: struct(nested_exprs)).alias(field.name))
                else:
                    # Simple field
                    df_field_name = df_normalized.get(field_lower, StructField("", StringType())).name
                    df_full_name = f"{prefix}{df_field_name}"
                    select_exprs.append(coalesce(col(f"`{df_full_name}`"), col(f"`{ice_full_name}`")).alias(field.name))
            else:
                # Field only in DataFrame
                df_field = df_normalized[field_lower]
                df_full_name = f"{prefix}{df_field.name}"

                if isinstance(field.dataType, StructType):
                    nested_exprs = build_select_expr(field.dataType, field.dataType, StructType([]), f"{df_full_name}.")
                    select_exprs.append(struct(nested_exprs).alias(field.name))
                elif (isinstance(field.dataType, ArrayType) and
                      isinstance(field.dataType.elementType, StructType)):
                    nested_exprs = build_select_expr(field.dataType.elementType, field.dataType.elementType,
                                                     StructType([]), "")
                    select_exprs.append(transform(col(df_full_name), lambda x: struct(nested_exprs)).alias(field.name))
                else:
                    select_exprs.append(col(f"`{df_full_name}`").alias(field.name))

        return select_exprs

    # Build merged schema
    merged_schema = build_merged_schema(df.schema, iceberg_schema)

    # Build select expressions
    select_expr = build_select_expr(merged_schema, df.schema, iceberg_schema)

    return df.select(select_expr)


# Example usage:
"""
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

spark = SparkSession.builder.appName("Test").getOrCreate()

# Sample DataFrame schema (after previous processing)
df_schema = StructType([
    StructField("First_Name", StringType(), True),
    StructField("User_Info", StructType([
        StructField("Age_Group", IntegerType(), True)
    ]), True),
    StructField("Extra_Field", StringType(), True)
])

# Sample Iceberg schema
iceberg_schema = StructType([
    StructField("first_name", StringType(), True),
    StructField("user_info", StructType([
        StructField("age_group", IntegerType(), True),
        StructField("new_field", StringType(), True)
    ]), True)
])

# Sample data
data = [
    ("John", {"Age_Group": 30}, "Extra"),
    ("Jane", {"Age_Group": 25}, "Data")
]

df = spark.createDataFrame(data, df_schema)

# Merge schemas
merged_df = merge_with_iceberg_schema(df, iceberg_schema)

merged_df.printSchema()
"""

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType
from pyspark.sql.functions import col, struct, transform


def merge_msg_payload_schema(df: DataFrame, golden_schema: StructType) -> DataFrame:
    """
    Merge the `msg_payload` field of a DataFrame's schema with a golden standard schema,
    adding new fields from the DataFrame while avoiding case-insensitive duplicates.
    The golden schema takes precedence for duplicate fields.

    Args:
        df: Input PySpark DataFrame with a `msg_payload` field
        golden_schema: Golden standard schema (e.g., from Iceberg table) with a `msg_payload` field
    Returns:
        DataFrame with merged schema for `msg_payload`
    """

    def normalize_field_name(name: str) -> str:
        """Normalize field name to lowercase for case-insensitive comparison."""
        return name.lower()

    def merge_schemas(df_schema: StructType, gold_schema: StructType, prefix: str = "", parent_array: bool = False):
        """
        Recursively merge two schemas, prioritizing golden schema for duplicates.

        Args:
            df_schema: DataFrame schema to merge
            gold_schema: Golden schema to prioritize
            prefix: Prefix for nested field paths
            parent_array: Flag indicating if schema is within an array
        Returns:
            Tuple of (merged schema, name mapping for select expressions)
        """
        gold_fields = {normalize_field_name(f.name): f for f in gold_schema.fields}
        df_fields = {normalize_field_name(f.name): f for f in df_schema.fields}
        new_fields = []
        name_mapping = {}  # Maps original DataFrame paths to new field names

        # First, include all golden schema fields
        for gold_field in gold_schema.fields:
            field_lower = normalize_field_name(gold_field.name)
            original_df_name = next(
                (name for name, field in df_fields.items() if normalize_field_name(name) == field_lower),
                gold_field.name)
            original_full_path = original_df_name if parent_array else f"{prefix}{original_df_name}"
            new_full_path = gold_field.name if parent_array else f"{prefix}{gold_field.name}"
            name_mapping[original_full_path] = new_full_path

            if isinstance(gold_field.dataType, StructType):
                # Merge nested struct
                df_nested_schema = df_fields.get(field_lower, StructField("", StructType([]))).dataType
                merged_nested_schema, nested_mapping = merge_schemas(df_nested_schema, gold_field.dataType,
                                                                     f"{gold_field.name}.", parent_array)
                new_field = StructField(gold_field.name, merged_nested_schema, gold_field.nullable)
                name_mapping.update(nested_mapping)
            elif isinstance(gold_field.dataType, ArrayType) and isinstance(gold_field.dataType.elementType, StructType):
                # Merge array of structs
                df_nested_schema = df_fields.get(field_lower,
                                                 StructField("", ArrayType(StructType([])))).dataType.elementType
                merged_nested_schema, nested_mapping = merge_schemas(df_nested_schema, gold_field.dataType.elementType,
                                                                     "", parent_array=True)
                new_field = StructField(gold_field.name, ArrayType(merged_nested_schema), gold_field.nullable)
                name_mapping.update(nested_mapping)
            elif isinstance(gold_field.dataType, MapType) and isinstance(gold_field.dataType.valueType, StructType):
                # Merge map with struct values
                df_nested_schema = df_fields.get(field_lower, StructField("", MapType(StringType(), StructType(
                    [])))).dataType.valueType
                merged_nested_schema, nested_mapping = merge_schemas(df_nested_schema, gold_field.dataType.valueType,
                                                                     "", parent_array=True)
                new_field = StructField(gold_field.name, MapType(gold_field.dataType.keyType, merged_nested_schema),
                                        gold_field.nullable)
                name_mapping.update(nested_mapping)
            else:
                # Simple field
                new_field = StructField(gold_field.name, gold_field.dataType, gold_field.nullable)

            new_fields.append(new_field)

        # Add non-duplicate DataFrame fields
        for df_field in df_schema.fields:
            field_lower = normalize_field_name(df_field.name)
            if field_lower not in gold_fields:
                original_full_path = df_field.name if parent_array else f"{prefix}{df_field.name}"
                new_full_path = df_field.name if parent_array else f"{prefix}{df_field.name}"
                name_mapping[original_full_path] = new_full_path

                if isinstance(df_field.dataType, StructType):
                    # Nested struct (no golden equivalent)
                    merged_nested_schema, nested_mapping = merge_schemas(df_field.dataType, StructType([]),
                                                                         f"{df_field.name}.", parent_array)
                    new_field = StructField(df_field.name, merged_nested_schema, df_field.nullable)
                    name_mapping.update(nested_mapping)
                elif isinstance(df_field.dataType, ArrayType) and isinstance(df_field.dataType.elementType, StructType):
                    # Array of structs (no golden equivalent)
                    merged_nested_schema, nested_mapping = merge_schemas(df_field.dataType.elementType, StructType([]),
                                                                         "", parent_array=True)
                    new_field = StructField(df_field.name, ArrayType(merged_nested_schema), df_field.nullable)
                    name_mapping.update(nested_mapping)
                elif isinstance(df_field.dataType, MapType) and isinstance(df_field.dataType.valueType, StructType):
                    # Map with struct values (no golden equivalent)
                    merged_nested_schema, nested_mapping = merge_schemas(df_field.dataType.valueType, StructType([]),
                                                                         "", parent_array=True)
                    new_field = StructField(df_field.name, MapType(df_field.dataType.keyType, merged_nested_schema),
                                            df_field.nullable)
                    name_mapping.update(nested_mapping)
                else:
                    # Simple field
                    new_field = StructField(df_field.name, df_field.dataType, df_field.nullable)

                new_fields.append(new_field)

        return StructType(new_fields), name_mapping

    def build_select_expr(merged_schema: StructType, name_mapping: dict, prefix: str = "", parent_array: bool = False):
        """
        Build select expressions to map DataFrame columns to merged schema.

        Args:
            merged_schema: Merged schema
            name_mapping: Mapping of original to new field paths
            prefix: Prefix for field names
            parent_array: Flag indicating if schema is within an array
        Returns:
            List of select expressions
        """
        exprs = []
        for field in merged_schema.fields:
            new_name = field.name
            full_new_name = new_name if parent_array else f"{prefix}{new_name}"
            original_full_name = next((k for k, v in name_mapping.items() if v == full_new_name), full_new_name)

            if isinstance(field.dataType, StructType):
                nested_prefix = "" if parent_array else f"{new_name}."
                nested_exprs = build_select_expr(field.dataType, name_mapping, nested_prefix, parent_array)
                exprs.append(struct(nested_exprs).alias(new_name))
            elif isinstance(field.dataType, ArrayType) and isinstance(field.dataType.elementType, StructType):
                nested_exprs = build_select_expr(field.dataType.elementType, name_mapping, "", parent_array=True)
                exprs.append(transform(col(f"`{original_full_name}`"), lambda x: struct(nested_exprs)).alias(new_name))
            elif isinstance(field.dataType, MapType) and isinstance(field.dataType.valueType, StructType):
                nested_exprs = build_select_expr(field.dataType.valueType, name_mapping, "", parent_array=True)
                exprs.append(transform(col(f"`{original_full_name}`"), lambda k, v: struct(
                    [k.alias("key"), struct(nested_exprs).alias("value")])).alias(new_name))
            else:
                exprs.append(col(f"`{original_full_name}`").alias(new_name))

        return exprs

    # Extract msg_payload schemas
    df_msg_schema = next((f.dataType for f in df.schema.fields if f.name.lower() == "msg_payload"), StructType([]))
    gold_msg_schema = next((f.dataType for f in golden_schema.fields if f.name.lower() == "msg_payload"),
                           StructType([]))

    # Merge msg_payload schemas
    merged_msg_schema, name_mapping = merge_schemas(df_msg_schema, gold_msg_schema)

    # Build new top-level schema
    new_fields = []
    for field in golden_schema.fields:
        if field.name.lower() == "msg_payload":
            new_fields.append(StructField(field.name, merged_msg_schema, field.nullable))
        else:
            new_fields.append(field)

    # Add non-duplicate top-level fields from DataFrame
    gold_fields = {f.name.lower(): f for f in golden_schema.fields}
    for field in df.schema.fields:
        if field.name.lower() not in gold_fields:
            new_fields.append(field)

    merged_schema = StructType(new_fields)

    # Build select expressions for top-level fields
    top_level_exprs = []
    for field in merged_schema.fields:
        if field.name.lower() == "msg_payload":
            msg_exprs = build_select_expr(merged_msg_schema, name_mapping)
            top_level_exprs.append(struct(msg_exprs).alias(field.name))
        else:
            top_level_exprs.append(col(f"`{field.name}`").alias(field.name))

    return df.select(top_level_exprs)
