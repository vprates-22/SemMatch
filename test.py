import json
import torch
import warnings
import numpy as np

from pathlib import Path
from functools import partial
from semmatch.evaluation.semantics import SemanticEval
# from semmatch.report.server import run_report

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings('ignore')

# Disable scientific notation
np.set_printoptions(suppress=True)

def print_fancy(d):
    print(json.dumps(d, indent=2))

if __name__ == "__main__":
    from reasoning.features.desc_reasoning import Reasoning, load_reasoning_from_checkpoint

    def match_reasoning_model(image0, image1, model):
        with torch.inference_mode():
            response = model.match({
                'image0': image0.unsqueeze(0).to(dev),
                'image1': image1.unsqueeze(0).to(dev),
            })
        
        mkpts0 = response['matches0'][0].detach().cpu().numpy()
        mkpts1 = response['matches1'][0].detach().cpu().numpy()
    
        return mkpts0, mkpts1

    reasoning_model_response = load_reasoning_from_checkpoint("DescriptorReasoning_ACCV_2024/models/xfeat", "checkpoint_2_1024000.pt")
    model = Reasoning(reasoning_model_response['model']).to(dev).eval()
    match_fn = partial(match_reasoning_model, model=model)

    semMatch = SemanticEval({
        'sam_model': 'sam2.1_l.pt',
        'data_path': 'hpatches',
        'dataset': 'hpatches',
        'max_pairs': 15
    })

    semMatch.extract_and_save_matches(match_fn, 'test')

    # run_report({
    #     'sam_model': 'sam2.1_l.pt',
    #     'matches_file_path': Path('SemMatch/output/matches/test_matches.npz'),
    #     'results_file_path': Path('SemMatch/output/reports/test_results.json')
    # }, port=6500)