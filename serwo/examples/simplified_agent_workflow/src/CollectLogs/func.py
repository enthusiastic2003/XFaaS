from azure.storage.queue import (
    QueueService,
    QueueMessageFormat
)
import json
from python.src.utils.classes.commons.serwo_objects import SerWOObject
import os, uuid


connect_str = "DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=xfaasstorage;AccountKey=K9juYVNksHxQzlooGhylToKxqmjr8LWn5SqcVb/tnajI9n6Is2LYbTJGqoEyEAfnmSUKyn0ACluq+AStJilx0Q==;BlobEndpoint=https://xfaasstorage.blob.core.windows.net/;FileEndpoint=https://xfaasstorage.file.core.windows.net/;QueueEndpoint=https://xfaasstorage.queue.core.windows.net/;TableEndpoint=https://xfaasstorage.table.core.windows.net/"
queue_name = 'xfaas-logging-queue-8pi0fc0'


queue_service = QueueService(connection_string=connect_str)

queue_service.encode_function = QueueMessageFormat.binary_base64encode
queue_service.decode_function = QueueMessageFormat.binary_base64decode


def function(serwoObject) -> SerWOObject:
    try:
        fin_dict = dict()
        data = serwoObject.get_body()
        print("Data to push - ", data)
        metadata = serwoObject.get_metadata()
        fin_dict['data'] = 'jpdc-executed-dummy-response'
        fin_dict['metadata'] = metadata
        print("Fin dict - ", fin_dict)
        queue_service.put_message(queue_name, json.dumps(fin_dict).encode('utf-8'))
        return SerWOObject(body=data)
    except Exception as e:
        return SerWOObject(error=True)
