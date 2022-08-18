import torch

from utils import text_cleaner
print(torch.cuda.is_available())

text = "Anh tư vấn rất nhiệt tình và kỹ lưỡng, còn kêu gọi mình cần gì cứ nhắn tin qua Fb hoặc Zalo cho anh, anh sẽ hổ trợ bất cứ thông tin gì mình cần. Nhiệt tình quá xá."
print(text_cleaner(text.lower()))