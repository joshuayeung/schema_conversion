from typing import Union
from pyiceberg.schema import Schema as IcebergSchema
from pyiceberg.types import (
    StructType as IcebergStructType, ListType as IcebergListType, MapType as IcebergMapType,
    NestedField, IntegerType, LongType, FloatType, DoubleType, StringType, BooleanType,
    BinaryType, TimestampType, TimestamptzType, DateType, DecimalType, UUIDType, TimeType
)
from pyspark.sql.types import (
    StructType, StructField, ArrayType, MapType, IntegerType as SparkIntegerType,
    LongType as SparkLongType, FloatType as SparkFloatType, DoubleType as SparkDoubleType,
    StringType as SparkStringType, BooleanType as SparkBooleanType, BinaryType as SparkBinaryType,
    TimestampType as SparkTimestampType, DateType as SparkDateType, DecimalType as SparkDecimalType
)

def iceberg_to_pyspark_schema(iceberg_schema: Union[IcebergSchema, IcebergStructType]) -> StructType:
    """
    Convert a pyiceberg schema to a PySpark schema.
    
    Args:
        iceberg_schema: Iceberg schema (pyiceberg Schema or StructType object)
        
    Returns:
        StructType: Equivalent PySpark schema
    """
    def _convert_type(iceberg_type: Union[IcebergStructType, IcebergListType, IcebergMapType, 
                                         IntegerType, LongType, FloatType, DoubleType, StringType, 
                                         BooleanType, BinaryType, TimestampType, TimestamptzType, 
                                         DateType, DecimalType, UUIDType, TimeType]) -> Union[
                                             StructType, ArrayType, MapType, SparkIntegerType, 
                                             SparkLongType, SparkFloatType, SparkDoubleType, 
                                             SparkStringType, SparkBooleanType, SparkBinaryType, 
                                             SparkTimestampType, SparkDateType, SparkDecimalType]:
        """Convert a pyiceberg type to a PySpark type."""
        if isinstance(iceberg_type, IcebergStructType):
            fields = [
                StructField(field.name, _convert_type(field.field_type), not field.required)
                for field in iceberg_type.fields
            ]
            return StructType(fields)
        elif isinstance(iceberg_type, IcebergListType):
            element_type = _convert_type(iceberg_type.element_type)
            return ArrayType(element_type, not iceberg_type.element_required)
        elif isinstance(iceberg_type, IcebergMapType):
            key_type = _convert_type(iceberg_type.key_type)
            value_type = _convert_type(iceberg_type.value_type)
            return MapType(key_type, value_type, not iceberg_type.value_required)
        elif isinstance(iceberg_type, (IntegerType, LongType, FloatType, DoubleType, StringType,
                                      BooleanType, BinaryType, TimestampType, TimestamptzType,
                                      DateType, TimeType, UUIDType)):
            type_mapping = {
                IntegerType: SparkIntegerType(),
                LongType: SparkLongType(),
                FloatType: SparkFloatType(),
                DecimalType: SparkDecimalType,
                DoubleType: SparkDoubleType(),
                StringType: SparkStringType(),
                BooleanType: SparkBooleanType(),
                BinaryType: SparkBinaryType(),
                TimestampType: SparkTimestampType(),
                TimestamptzType: SparkTimestampType(),
                DateType: SparkDateType(),
                TimeType: SparkLongType(),  # TimeType maps to LongType (microseconds since midnight)
                UUIDType: SparkStringType(),
            }
            if isinstance(iceberg_type, DecimalType):
                return SparkDecimalType(precision=iceberg_type.precision, scale=iceberg_type.scale)
            return type_mapping.get(type(iceberg_type), SparkStringType())  # Fallback to StringType
        else:
            return SparkStringType()  # Fallback for unhandled types

    # If input is an Iceberg Schema, get its StructType
    if isinstance(iceberg_schema, IcebergSchema):
        iceberg_struct = iceberg_schema.as_struct()
    else:
        iceberg_struct = iceberg_schema

    if not isinstance(iceberg_struct, IcebergStructType):
        raise ValueError("Iceberg schema must be a Schema or StructType")

    return _convert_type(iceberg_struct)
