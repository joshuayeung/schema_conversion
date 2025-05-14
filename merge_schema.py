from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, ArrayType, MapType, StringType, IntegerType
from pyspark.sql.functions import col, struct, transform, coalesce


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
