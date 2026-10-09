import timm
 
# timm.create_model('vit_large_patch14_reg4_dinov2.lvd142m', pretrained=True)
timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True)

print('Model cached successfully')