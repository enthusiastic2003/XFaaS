from python.src.utils.classes.commons.serwo_objects import SerWOObject
from groq import Groq
import json
import logging

def function(serwoObject ) -> SerWOObject:
    try:
        client = Groq(api_key="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")

        body = serwoObject.get_body()
        metadata = serwoObject.get_metadata()

        logging.info("Received body for itinerary generation: %s", body)

        # Extract and validate 'place'
        place_input = body.get("places", ["Eiffel Tower"])
        if isinstance(place_input, list) and place_input:
            place = place_input[1]
        elif isinstance(place_input, str):
            place = place_input
        else:
            raise ValueError("Invalid or missing 'place' in input. Must be a string or non-empty list.")

        itinerary_prompt = (
            f"Plan a 1-day itinerary for visiting {place} in India. "
            f"Be as compact as possible. Only provide a list format itinerary for: "
            f"1. Morning 2. Afternoon 3. Evening 4. Night."
        )

        logging.info(f"Prompting Groq for itinerary of: {place}")
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": itinerary_prompt}]
        )

        itinerary = response.choices[0].message.content
        logging.info(f"Received itinerary for {place}: {itinerary}")

        # Append new itinerary
        if "split_iterary" not in body: 
            body["split_iterary"] = []

        body["split_iterary"].append((place, itinerary))  # store as tuple for later merging

        logging.info(f"Updated body with itinerary: {str(body)}")
        newbody =  SerWOObject(body=body, metadata=metadata)
        newbody.set_basepath(serwoObject.get_basepath())

        return newbody

    except Exception as e:
        logging.info("Exception in itinerary function: %s", str(e))
        return SerWOObject(error={"message": str(e)})
