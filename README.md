# Changes

- Added `--full_folder` param to `eval_linear`
    - Take the entire "linear_prob_dataset" and does a 80/20 split into train and val in `set_loaders` line 90
- Added `label_to_supcat` and `set_meters`
- Added `best_acc` map in main for INat, be sure to change if needed for ORE
- Changes to `_average_binary_score` to handle general lable_to_groups

- Commented out `high_resolution` and `low_resolutions` pic handling in `train` method, besure sure to add that in and handle properly. Such as at line 486: 

```python
            # low_resolution_output = model(low_resolution_batch)
            # high_resolution_output = model(high_resolution_batch)
            # output = torch.cat((high_resolution_output, low_resolution_output), dim=0)
            output = model(image_list)
```

