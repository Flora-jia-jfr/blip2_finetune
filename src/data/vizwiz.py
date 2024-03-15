import torch
import os
import json
from collections import defaultdict, Counter
from torch.utils.data import Dataset
from PIL import Image
import pickle as pkl
import random
import logging
from torchvision import transforms as T
from tqdm import tqdm

from utils.vqa_utils import get_score

logger = logging.getLogger(__name__)
logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)

class VizwizDataset(Dataset):
    '''
        use the most-selecter answer for training
        evaluate the generated answer based on the get_score function (analyze over all the answers)
    '''

    def __init__(self, split, mode='q2a', vis_processors=None, text_processors=None, demo=False):

        data_dir = '/home/shared/MCL/vizwiz'
        images_dir = os.path.join('/home/shared/MCL/vizwiz/images', split)
        image_filenames = os.listdir(images_dir)
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.mode = mode
        self.split = split
        self.demo = demo
        self.qid2score_dict = {}

        if vis_processors is not None:
            self.vis_processor = vis_processors['train'] if split == 'train' else vis_processors['eval']
        if text_processors is not None:
            self.text_processor = text_processors['train'] if split == 'train' else text_processors['eval']

        self.imageid2filename = {}
        # for fn in image_filenames:
        #     image_id = int(fn.split('_')[-1].strip('.jpg'))
        #     self.imageid2filename[image_id] = os.path.join(images_dir, fn)
        # self.imageids = list(set(list(self.imageid2filename.keys())))

        self.annotations_file = os.path.join(data_dir, 'Annotations/{}.json'.format(split))

        # for preprocess dataset
        if self.demo:
            self.cached_data_dir = os.path.join(data_dir, f'cached_vizwiz_demo_data_for_blip')
        else:
            self.cached_data_dir = os.path.join(data_dir, f'cached_vizwiz_data_for_blip')

        os.makedirs(self.cached_data_dir, exist_ok=True)

        self.cached_data_file = os.path.join(self.cached_data_dir, 'vizwiz_preprocessed_{}.pkl'.format(split))
        self.cached_qid2score_dict_file = os.path.join(self.cached_data_dir, 'qid2score_dict_{}.pkl'.format(split))

        if os.path.exists(self.cached_data_file) and os.path.exists(self.cached_qid2score_dict_file):
            self.data = pkl.load(open(self.cached_data_file, 'rb'))
            self.qid2score_dict = pkl.load(open(self.cached_qid2score_dict_file, 'rb'))
            annotations = json.load(open(self.annotations_file))
            self.imageid2filename = {image_id: os.path.join(self.images_dir, anno['image']) for image_id, anno in enumerate(annotations)}
            logger.info("Loaded Vizwiz {} dataset, with {} examples".format(self.split, len(self.data)))
        else:
            annotations = json.load(open(self.annotations_file))
            self.data = []
            for idx, anno in enumerate(annotations):
                if self.demo and idx == 1000:
                    break
                qid = idx
                image_id = idx
                self.imageid2filename[image_id] = os.path.join(self.images_dir, anno['image'])

                # Retrieve the question for this annotation
                question = anno['question']

                # Map from each crowdsourced answer to occurrences in annotation
                answers = [a['answer'] for a in anno['answers']]
                answer_count = defaultdict(int)
                for ans in answers:
                    answer_count[ans] += 1

                # Get score (0.3/0.6/1) corresponding to each crowdsourced answer
                scores = {}
                answers = []
                top_answer_cnt = 0
                top_answer = None
                for answer in answer_count:
                    score = get_score(answer_count[answer])
                    scores[answer] = score
                    answers.append(answer)
                    if answer_count[answer] > top_answer_cnt:
                        top_answer_cnt = answer_count[answer]
                        top_answer = answer
                    # elif answer_count[answer] == top_answer_cnt:
                    #     print(f"equal count between {answer} and {top_answer}: {answer_count[answer]} = {top_answer_cnt}")
                assert top_answer is not None
                self.qid2score_dict[qid] = scores

                # Store pre-processed example
                example = {
                    'question_id': qid,
                    'image_id': image_id,
                    'question': question,
                    'answers': answers,
                    'scores': scores,
                    'top_answer': top_answer,
                }
                self.data.append(example)

            pkl.dump(self.data, open(self.cached_data_file, 'wb'))
            pkl.dump(self.qid2score_dict, open(self.cached_qid2score_dict_file, 'wb'))

            logger.info("Created and Loaded Vizwiz {} dataset, with {} examples".format(self.split, len(self.data)))

        if vis_processors is None or text_processors is None:
            logger.warning("Vision/text processors not set!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]
        image_id = data['image_id']
        assert image_id in self.imageid2filename.keys()
        image_fn = self.imageid2filename[image_id]
        raw_image = Image.open(image_fn).convert('RGB')
        
        question = data['question']
        answer = data['top_answer']
        if self.mode == 'q2a':
            text_input = f"Question: {question} Answer: "
            text_output = answer
        else:
            raise NotImplementedError(f"{self.mode} not implemented for VQA dataset")

        score_dict = data['scores']
        image_id = data['image_id']
        qid = data['question_id']

        output = {
            'text_input': text_input,
            'text_output': text_output,
            'raw_image': raw_image,
            'score_dict': score_dict,
            'image_path': image_fn,
            'image_id': image_id,
            'question': question,
            'top_answer': answer, # added
            'qid': qid
        }
        return output

    def task_collate_fn(self, batch):

        images = [b['raw_image'] for b in batch]
        processed_images = torch.stack([self.vis_processor(img) for img in images], dim=0)

        text_inputs = [b['text_input'] for b in batch]
        processed_text_inputs = [self.text_processor(txt) for txt in text_inputs]

        text_outputs = [b['text_output'] for b in batch]
        try:
            processed_text_outputs = [self.text_processor(str(txt)) for txt in text_outputs]
        except AttributeError as e:
            print(e)
            print("text_inputs: ", text_inputs)
            print("text_outputs: ", text_outputs)
            # contain numbers
            exit()

        score_dict = [b['score_dict'] for b in batch]
        qids = [b['qid'] for b in batch]

        collated_batch = {
            "image": processed_images,
            "text_input": processed_text_inputs,
            "text_output": processed_text_outputs,
            "prompt": text_inputs,
            "target": text_outputs,
            "qids": qids,
        }
        return collated_batch

if __name__ == '__main__':
    dataset = VizwizDataset('train')
    # import pdb; pdb.set_trace()
