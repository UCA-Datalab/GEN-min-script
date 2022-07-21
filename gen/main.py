import typer
from gen.data_processing import process_pdf
from gen.models import inference 


def generate_results(pdf_bytes):
    images, texts = process_pdf(pdf_bytes)

    results = inference(images,texts)

    return results

    

# ESTA FUNCIÓN ES LA ENTRADA DEL SCRIPT
def main(pdf_input_path: str,csv_output_path:str):
    # El pdf se abre en bytes
    with open(pdf_input_path, "rb") as f:
        pdf_bytes = f.read()

    # Se hace la predicción y devuelve un DataFrame de pandas
    results_df = generate_results(pdf_bytes)

    results_df.to_csv(csv_output_path)

if __name__ == "__main__":
    typer.run(main)
