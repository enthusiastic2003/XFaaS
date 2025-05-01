from python.src.utils.classes.commons.serwo_objects import SerWOObject
from groq import Groq
import json
import logging

def function(serwoObject ) -> SerWOObject:
    try:
        client = Groq(api_key="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")

        body = serwoObject.get_body()
        place = body.get("place", "World")

        places_prompt = (
            f"You are an API returning data. Respond with ONLY a valid JSON object and nothing else. "
            f"Suggest exactly 3 must-visit place in {place}. "
            f"The JSON must have one key: 'places', and the value must be a list containing 3 strings. "
            f"Example: {{\"places\": [\"Taj Mahal\", \"Victoria Memorial\", \"Aligarh Zoo\"]}}. Do not include explanations, formatting, or any additional text."
        )


        logging.info(f"[Step 1] Prompting Groq for place suggestions for: {place}")

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": places_prompt}]
        )

        places_response = response.choices[0].message.content
        logging.info(f"[Step 2] Received response: {places_response}")

        places_json = json.loads(places_response)
        logging.info(f"[Step 2] Parsed JSON: {str(places_json)}")
        places_json['split_iterary'] = []

        logging.info(f"[Step 3] Final JSON to pass forward: {str(places_json)}")

        newbody =  SerWOObject(
            body=places_json,
            metadata=serwoObject.get_metadata()
        )

        newbody.set_basepath(serwoObject.get_basepath())

        return newbody

    except Exception as e:
        logging.info("Exception in place-suggestion function: " + str(e))
        return SerWOObject(error={"message": str(e)})
