import typer
from gen.data_processing import process_pdf
from gen.models.inference import inference

def generate_results(pdf_bytes):
    images, texts = process_pdf(pdf_bytes)

    results_labeled = inference(images, texts)

    return results_labeled


# ESTA FUNCIÓN ES LA ENTRADA DEL SCRIPT
def main(pdf_input_path: str):
    # El pdf se abre en bytes
    with open(pdf_input_path, "rb") as f:
        pdf_bytes = f.read()

    # Se hace la predicción y devuelve una lista
    results = generate_results(pdf_bytes)
    print(results)


if __name__ == "__main__":
    typer.run(main)
