# permeability_reconstruction_network

To get started, first install the dependencies:
```console
pip3 install -r requirements_without_torch.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Optional:
```console
pip3 install wandb 
```

You can download data samples and the model state from the following URLs:

*model weigths -*
https://huggingface.co/datasets/mexalon/microseicmic_events_to_permeability_map/resolve/main/permnet_wights.pt

*train dataset -*
https://huggingface.co/datasets/mexalon/microseicmic_events_to_permeability_map/resolve/main/train_mini_set.h5

*test dataset -*
https://huggingface.co/datasets/mexalon/microseicmic_events_to_permeability_map/resolve/main/test_mini_set.h5

*sample permeability models -*
https://huggingface.co/datasets/mexalon/microseicmic_events_to_permeability_map/resolve/main/permeability_models.h5

or you can run the corresponding cells in notebooks.
