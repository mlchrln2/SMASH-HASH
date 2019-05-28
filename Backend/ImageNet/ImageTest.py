'''Main method for validating the results once a model is deployed at the end of every
training epoch'''

# dependencies
import gc
import sys
import torch
from tensorboardX import SummaryWriter

# modules used for testing and viewing
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# user defined modules
from DataLoader import TEST_LOADER as dataloader
from DataLoader import IDX2WORD


# set up a method for drawing the images
TO_IMG = ToPILImage()


def plot(pics, alps, caption, caps, fig_num):
    '''A method for plotting the image locations attended to by the model and the corresponding
    captions.'''
    alps = alps.squeeze(0)
    width = alps.size(0)
    fig = plt.figure(fig_num, figsize=(10, width))
    fig.add_subplot(2, 1, 1)
    pics = TO_IMG(pics.squeeze(0))
    plt.title(" ".join(caps))
    plt.imshow(pics)
    for k in range(width):
        fig.add_subplot(2, width, k + width + 1)
        plt.title(caption[k])
        plt.imshow(pics)
        alpha = TO_IMG(alps[k].unsqueeze(0))
        plt.imshow(alpha, cmap='gray', alpha=.7)
    plt.show()
    WRITER.add_figure('plot_{}'.format(fig_num), fig, fig_num, True)
    plt.close()

# initialize model and loss function
CHECKPOINT = torch.load(sys.argv[1], map_location='cpu')
MODEL = CHECKPOINT['model']
INFERENCE = {'greedy': MODEL.infer_greedy_search, 
             'beam': MODEL.infer_beam_search}[sys.argv[2]]
# print('Note model parameters:\n{}'.format(MODEL.parameters))

# set the mode to train
MODEL.eval()

# create a logger
WRITER = SummaryWriter()

'''
WRITER.add_graph(MODEL, (torch.randn(1, 3, 224, 224), torch.randint(20, (1, 20), dtype=torch.long)),
				 vervose=True)
'''

gc.collect()
# inference method that tests the top-k captions for the model
for i, (image, img, labels, lengths) in enumerate(dataloader):
    labels = labels.squeeze(0)[1:-1]
    all_words, all_summaries, all_alphas = INFERENCE(img)
    for j, _ in enumerate(all_words):
        words, summaries, alphas = all_words[
            j], all_summaries[j].unsqueeze(0), all_alphas[j]
        print(words.size(), summaries.size(), alphas.size())
        sentence = [IDX2WORD[str(word.item())].value.decode(
            "utf-8") for word in words]
        phrase = [IDX2WORD[str(word.item())].value.decode("utf-8")
                  for word in labels]
        plot(img, alphas, sentence, phrase, i)
    print('iteration {} of {}'.format(i + 1, len(dataloader)), end='\r')
