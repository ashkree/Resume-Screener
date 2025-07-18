def process_resume(parsed_text):

    snippet = parsed_text[:150].replace("\n", " ")
    return f"AI Review Summary: Your CV contains {len(parsed_text)} characters. Preview: {snippet}..."
