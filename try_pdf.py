import bet.sample as samp
from datetime import datetime
from bet.util import comm
startTime0 = datetime.now()

print('sample generation')
startTime = datetime.now()
s = samp.sample_set(1)
s.set_dist()
s.generate_samples(1E7, globalize=True)
print(datetime.now() - startTime)

globalize = True

if comm.rank == 0 or comm.rank == comm.size-1:
    print('comm rank: %d, %d'%(comm.rank, 
            s.check_num_local()))
    try: 
        print(s._values.shape)
    except:
        print(s._values_local.shape)

if globalize: 
    print('local to global')
    startTime = datetime.now()
    s.local_to_global()
    print(datetime.now() - startTime)
    print('\t KDE construction')
    startTime = datetime.now()
    from scipy.stats import gaussian_kde as gkde
    s._distribution = gkde(s.get_values().T)
    print(datetime.now() - startTime)

print('pdf computation')
# time the pdf computation
startTime = datetime.now()
p = s.pdf(s.get_values()[0:100,:])
print(datetime.now() - startTime)


# # print(p[0:10])
# print(p.shape)

if comm.rank == 0:
    print('\t FINAL:', datetime.now() - startTime0)
