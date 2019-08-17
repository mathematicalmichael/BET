import bet.sample as samp
from datetime import datetime


s = samp.sample_set(2)
s.set_dist()
s.generate_samples(1E8)
print(s.check_num_local())

# time the pdf computation
startTime = datetime.now()
p = s.pdf(s.get_values())
print(datetime.now() - startTime)

# print(p[0:10])
print(p.shape)

