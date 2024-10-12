import os
from deepface import DeepFace
from sklearn.metrics import f1_score, classification_report

# Caminho para o diretório com as pastas das celebridades (treino)
celebrity_dir = r'C:\Doutorado\BdCelebridades2\archive\data\train'  # Substitua pelo caminho correto

# Caminho para o diretório com as imagens de teste
test_dir = 'C:/Doutorado/BdCelebridades2/archive/data/val'  # Substitua pelo caminho correto

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

# Listas para armazenar as previsões e os rótulos reais
true_labels = []
predicted_labels = []

# Iterar sobre as imagens de teste
for celeb_folder in os.listdir(test_dir):
    celeb_test_path = os.path.join(test_dir, celeb_folder)
    
    if os.path.isdir(celeb_test_path):
        for img_file in os.listdir(celeb_test_path):
            test_image_path = os.path.join(celeb_test_path, img_file)

            # Chamar a função para prever a celebridade
            predicted_label, score = find_closest_match(test_image_path, celebrity_dir)

            # Adicionar o rótulo verdadeiro e o previsto às listas
            true_labels.append(celeb_folder)  # O nome da pasta é o rótulo real
            predicted_labels.append(predicted_label)

# Calcular o F1-Score
f1 = f1_score(true_labels, predicted_labels, average='weighted')  # 'weighted' para lidar com classes desbalanceadas

# Exibir o F1-Score e o relatório de classificação completo
print(f'F1-Score: {f1}')
print(classification_report(true_labels, predicted_labels))
