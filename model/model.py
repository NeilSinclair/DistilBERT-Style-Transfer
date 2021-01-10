def freeze_params(model, action = "freeze"):
  ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
      adapted from finetune.py '''
  for layer in model.parameters():
    if action == "freeze":
      layer.requires_grad = False
    else:
      layer.requires_grad = True

def freeze_multiple_params(model_method, params):
  '''
    :param model_method: list containing the model methods (e.g. model.distilbert.embeddings.word_embeddings)
                         which should be frozen or not
    :param params: list of hparams indicating which of the above should be frozen or not
  '''
  for i, param in enumerate(params):
    if param:
      freeze_params(model_method[i])
    elif param == False:
      freeze_params(model_method[i], action = "no_freeze")