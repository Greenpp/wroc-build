# %%
from wroclaw_building_footprint import Segmentator
# %%
seg = Segmentator()
# %%
result = seg.segment(51.098337241783774, 17.070753329397945)
# %%
prob = result.get_building_probability()
print(f'Probability of a building: {prob * 100:.2f}%')
# %%
result.show_img()
# %%
result.show_probability_mask()
# %%
result.show_binary_mask()
# %%
