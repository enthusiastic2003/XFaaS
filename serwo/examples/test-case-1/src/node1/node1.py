from python.src.utils.classes.commons.serwo_objects import SerWOObject

def function(serwoObject) -> SerWOObject:
    try:
        
        body = str(serwoObject.get_body())

        print('xml end')
        ret_val = "ECHO BACK 1: " + body
        s = SerWOObject(body=ret_val)
        return s
    except Exception as e:
        print('in xml '+e)
        return None