import os
from deepface import DeepFace

# Caminho para o diretório com as pastas das celebridades
celebrity_dir = r'C:\Doutorado\BdCelebridades2\archive\data\train' # Substitua pelo caminho do dataset "5 Celebrity Faces"

# Função para encontrar a correspondência mais próxima
def find_closest_match(test_image_path, celebrity_dir):
    best_match = None
    best_score = float('inf')  # Inicialmente, definimos a menor distância como infinita

    # Iterar sobre as pastas de cada celebridade
    for celeb_folder in os.listdir(celebrity_dir):
        celeb_path = os.path.join(celebrity_dir, celeb_folder)
        
        if os.path.isdir(celeb_path):  # Se for uma pasta de celebridade
            for img_file in os.listdir(celeb_path):
                img_path = os.path.join(celeb_path, img_file)
                
                # Comparar a imagem de teste com a imagem atual da pasta
                result = DeepFace.verify(test_image_path, img_path, enforce_detection=False)
                distance = result['distance']

                # Verificar se a distância é menor que a atual melhor
                if distance < best_score:
                    best_score = distance
                    best_match = celeb_folder

    # Retornar o nome da celebridade com a melhor correspondência
    return best_match, best_score

# Escolher a imagem de teste
test_image_path = r'C:\Doutorado\BdCelebridades2\archive\data\val\ben_afflek\httpcsvkmeuadecafjpg.jpg'  # Caminho para a imagem de teste que você deseja identificar

# Executar a função para encontrar a melhor correspondência
celebrity_name, score = find_closest_match(test_image_path, celebrity_dir)
print(f'A imagem corresponde à celebridade: {celebrity_name} com uma pontuação de similaridade: {score}')
