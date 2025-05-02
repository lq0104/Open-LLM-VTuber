# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import mimetypes
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")


def generate(prompt, api_key=None):
    """
    使用Gemini API生成图像
    
    Args:
        prompt (str): 图像生成提示
        api_key (str, optional): Gemini API密钥，如果为None则使用默认密钥
    """
    client = genai.Client(
        # vertexai=True,
        project="",
        location="",
        api_key=api_key,
    )

    model = "gemini-2.0-flash-exp-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "image",
            "text",
        ],
        response_mime_type="text/plain",
    )

    file_extension = ""
    chunk_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
        else:
            chunk_text += chunk.text
            
    return data_buffer,file_extension,chunk_text

if __name__ == "__main__":
    prompt = "a beautiful lake"
    data_buffer, file_extension, chunk_text = generate(prompt)
    file_name = "test"
    save_binary_file(f"{file_name}{file_extension}", data_buffer)
    print(f"generate image chunk_text: {chunk_text}")