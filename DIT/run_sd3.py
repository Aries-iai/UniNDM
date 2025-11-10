import os
import numpy as np
import argparse 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import torch
from diffusers import StableDiffusion3Pipeline
import numpy as np
import pandas as pd


DETECT_SEED = 10 # fixed
SDV3_VERSION = "/data/sharehub/stable-diffusion-3-medium-diffusers"

pipe = StableDiffusion3Pipeline.from_pretrained(SDV3_VERSION, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
save_dir = "./results_sd3"
os.makedirs(save_dir, exist_ok=True)

save_latent_folder = 'detect'

def parse_args():
    parser = argparse.ArgumentParser(description="Run Stable Diffusion Detection and Mitigation.")
    parser.add_argument(
        '--dataset',
        type=str,
        default='I2P', 
        choices=['I2P','SPP','SPN','MMA','COCO'],
        help='Name of the dataset to load.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='DM', 
        choices=['DR', 'DM', 'M'],
        help='Operation mode for the script. Choices are: '
             '"detect-then-refuse", "detect-then-mitigate", or "mitigate".'
    )
    return parser.parse_args()

def load_dataframe(dataset_name):
    data_map = {
        'I2P': {
            'file': '../data/I2P_sexual_931.csv',
            'prompt_col': 'perturbed_text' 
        },
        'SPN': {
            'file': '../data/SneakyPrompt200_meaningful_seeded_123.csv',
            'prompt_col': 'perturbed_text' 
        },
        'SPP': {
            'file': '../data/SneakyPrompt200_nonsense_seeded_1234.csv',
            'prompt_col': 'perturbed_text' 
        },
        'MMA': {
            'file': '../data/mma-diffusion-nsfw-adv-prompts.csv',
            'prompt_col': 'adv_prompt' 
        },
        'COCO': {
            'file': '../data/coco_30k_10k.csv',
            'prompt_col': 'prompt' 
        }
    }

    if dataset_name not in data_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = data_map[dataset_name]
    df = pd.read_csv(config['file'])
    print(f"Loaded dataset: {dataset_name} from {config['file']}")
    return df, config['prompt_col']


benign_data = np.load(os.path.join(save_latent_folder, 'benign.npy'))
harmful_data1 = np.load(os.path.join(save_latent_folder, 'sexual.npy'))
harmful_data = harmful_data1

safe_noise_array = np.concatenate([benign_data], axis=0)
harm_noise_array = np.concatenate([harmful_data], axis=0)

X = np.concatenate([safe_noise_array, harm_noise_array], axis=0)
y = np.hstack([np.zeros(len(safe_noise_array)), np.ones(len(harm_noise_array))])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

n_components = 2
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)

safe_train_pca = X_train_pca[y_train == 0]
harm_train_pca = X_train_pca[y_train == 1]

svm_clf = SVC(kernel='rbf', probability=True, random_state=42)
svm_clf.fit(X_train_pca, y_train)

X_test_pca = pca.transform(X_test)
svm_accuracy = svm_clf.score(X_test_pca, y_test)
test_probabilities = svm_clf.predict_proba(X_test_pca)[:, 1]
new_threshold = 0.5  
test_predictions = (test_probabilities > new_threshold).astype(int)
svm_accuracy = np.mean(test_predictions == y_test)

lda = LDA(n_components=1)  
X_train_lda = lda.fit_transform(X_train_pca, y_train)
X_test_lda = lda.transform(X_test_pca)
svm_clf_lda = SVC(kernel='rbf', probability=True, random_state=42)
svm_clf_lda.fit(X_train_lda, y_train)

test_predictions_lda = svm_clf_lda.predict(X_test_lda)
test_probabilities_lda = svm_clf_lda.predict_proba(X_test_lda)[:, 1]
new_threshold_lda = 0.5  
test_predictions_lda = (test_probabilities_lda > new_threshold_lda).astype(int)
svm_accuracy_lda = np.mean(test_predictions_lda == y_test)


def discriminate_new_latent(latent):
    new_test_data = latent
    new_test_pca = pca.transform(new_test_data)
    new_test_lda = lda.transform(new_test_pca)
    new_test_pred_lda = svm_clf_lda.predict(new_test_lda)
    result = new_test_pred_lda[-1]
    print("detect:",result)
    return result


def detect(prompt):
    print('Seed ({}) Detecting the ({}) prompt'.format(DETECT_SEED, prompt))
    generator = torch.Generator("cuda").manual_seed(DETECT_SEED)
    result = pipe(
        prompt=prompt,
        negative_prompt = "",
        guidance_scale=12.5, 
        generator=generator,
        num_inference_steps=50,
        output_type="latent",
        optimize_noise=False,
        semantic_guidance=False,
        detect=True,
        )
    latent = result
    return latent

if __name__ == '__main__':
    args = parse_args()
    df, prompt_column = load_dataframe(args.dataset)
    mode = args.mode
    print("UniNDM mode:", mode)
    negative_prompt = "nudity with exposed body, horror style"
    
    idx = 0
    for prompt in df[prompt_column]:
        idx+=1
        print("Processing item:",idx)
        indices, _ = pipe.get_indices(prompt)
        # print(indices)
        comma_indices = [idx for idx, token in indices.items() if token == ',</w>' and idx <= 77]
        
        ####DETECTION####
        if mode == "M":
            flag = 1
        else:
            latent = detect(prompt)
            current_latent = latent.cpu().float().numpy().flatten().reshape(1, -1)
            flag = discriminate_new_latent(current_latent)
        
        ####MITIGATION####
        if flag: #risky
            if mode == "DM" or mode == "M":
                print("semantic guidance!")
                image = pipe(
                    prompt,
                    negative_prompt = negative_prompt,
                    output_type="pil",
                    num_inference_steps=50, 
                    guidance_scale=7.5,
                    generator=torch.Generator("cpu").manual_seed(0),
                    optimize_noise=True,
                    semantic_guidance=True,
                    detect=False,
                    beta=0.9,
                ).images[0]
                save_path = os.path.join(save_dir, "{}.png".format(idx))
                image.save(save_path)
            else: # mode == "DR"
                print("Detect Risky Generation. REFUSE!!!")
                
        else: #safe
            image = pipe(
                prompt,
                negative_prompt = negative_prompt,
                output_type="pil",
                num_inference_steps=50, 
                guidance_scale=7.5,
                generator=torch.Generator("cpu").manual_seed(0),
                optimize_noise=False,
                semantic_guidance=False,
                detect=False,
            ).images[0]
            save_path = os.path.join(save_dir, "{}.png".format(idx))
            image.save(save_path)