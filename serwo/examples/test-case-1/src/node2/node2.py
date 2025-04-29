from python.src.utils.classes.commons.serwo_objects import SerWOObject

from python.src.utils.classes.commons.serwo_objects import SerWOObject

def function(serwoObject) -> SerWOObject:
    try:
        # Extract input payload
        body = serwoObject.get_body()
        name = "stranger"

        # Body could be str or dict (depending on curl input)
        if isinstance(body, dict) and "name" in body:
            name = body["name"]
        elif isinstance(body, str):
            try:
                import json
                body_json = json.loads(body)
                name = body_json.get("name", "stranger")
            except:
                pass  # fallback to default

        # Compose response
        message = f"Assalamalekim, {name}!"

        # Construct return object
        return_object = SerWOObject(
            body={"message": message},
            metadata=serwoObject.get_metadata()
        )
        return_object.set_basepath(serwoObject.get_basepath())

        return return_object

    except Exception as e:
        print("Exception in greeter:", str(e))
        return SerWOObject(error={"message": str(e)})
