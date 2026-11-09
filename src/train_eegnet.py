"""Fixed EEGNet baseline: A01-A07 development, A08-A09 one-time final test."""
import argparse, json, random, time, sys, traceback
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from src.load_data import load_split_data, DATASET_ARCHIVES, _get_processed_dir
from src.models.eegnet import EEGNet

def metrics(y, p):
    a,b,c,d=precision_recall_fscore_support(y,p,labels=[0,1,2,3],zero_division=0)
    return {"accuracy":float(accuracy_score(y,p)),"macro_f1":float(c.mean()),"per_class":{str(i):{"precision":float(a[i]),"recall":float(b[i]),"f1":float(c[i]),"support":int(d[i])} for i in range(4)},"confusion_matrix":confusion_matrix(y,p,labels=[0,1,2,3]).tolist()}
def loader(x,y,b,shuffle=False): return DataLoader(TensorDataset(torch.from_numpy(x[:,None]).float(),torch.from_numpy(y).long()),batch_size=b,shuffle=shuffle,num_workers=0)
def evaluate(model, dl, device):
    model.eval(); ys=[]; ps=[]; loss=0.; n=0
    with torch.no_grad():
      for x,y in dl:
        x,y=x.to(device),y.to(device); z=model(x); loss+=torch.nn.functional.cross_entropy(z,y).item()*len(y); n+=len(y); ys.append(y.cpu().numpy()); ps.append(z.argmax(1).cpu().numpy())
    return loss/n,np.concatenate(ys),np.concatenate(ps)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--max-epochs',type=int,default=150);p.add_argument('--patience',type=int,default=20);p.add_argument('--resume',action='store_true');p.add_argument('--validation-only',action='store_true');p.add_argument('--temporal-filters',type=int,default=8);p.add_argument('--depth-multiplier',type=int,default=2);p.add_argument('--dropout',type=float,default=.5);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--reduce-lr-on-plateau',action='store_true');p.add_argument('--label-smoothing',type=float,default=0.0);p.add_argument('--normalization',choices=('batch','group'),default='batch');a=p.parse_args()
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.use_deterministic_algorithms(True,warn_only=True);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 xtr,ytr,_=load_split_data('train',window='central',dataset='cross_subject');xv,yv,_=load_split_data('val',window='central',dataset='cross_subject')
 model=EEGNet(temporal_filters=a.temporal_filters,depth_multiplier=a.depth_multiplier,dropout=a.dropout,normalization=a.normalization).to(device);opt=torch.optim.Adam(model.parameters(),lr=a.lr,weight_decay=a.weight_decay);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=8) if a.reduce_lr_on_plateau else None;start_epoch=1;best=(-1.,float('inf'),0);history=[];stale=0
 if a.resume:
  resume_path=a.output_dir/'last.pt' if (a.output_dir/'last.pt').exists() else a.output_dir/'best_validation.pt';ck=torch.load(resume_path,map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict']);
  if 'optimizer_state_dict' in ck: opt.load_state_dict(ck['optimizer_state_dict'])
  start_epoch=ck['epoch']+1;history=[json.loads(x) for x in (a.output_dir/'history.jsonl').read_text().splitlines()];best_ck=torch.load(a.output_dir/'best_validation.pt',map_location='cpu',weights_only=False);best=(best_ck['val_metrics']['macro_f1'],float('inf'),best_ck['epoch'])
 else:
  a.output_dir.mkdir(parents=True,exist_ok=False);(a.output_dir/'config.json').write_text(json.dumps({'seed':42,'batch_size':32,'lr':a.lr,'weight_decay':a.weight_decay,'temporal_filters':a.temporal_filters,'depth_multiplier':a.depth_multiplier,'dropout':a.dropout,'normalization':a.normalization,'label_smoothing':a.label_smoothing,'reduce_lr_on_plateau':a.reduce_lr_on_plateau,'max_epochs':a.max_epochs,'patience':a.patience,'dataset':'cross_subject','window':'central','validation_only':a.validation_only},indent=2))
 start=time.perf_counter()
 for epoch in range(start_epoch,a.max_epochs+1):
  model.train();total=n=0
  for x,y in loader(xtr,ytr,32,True):
   x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(x),y,label_smoothing=a.label_smoothing);loss.backward();opt.step();total+=loss.item()*len(y);n+=len(y)
  vl,yt,yp=evaluate(model,loader(xv,yv,32),device);vm=metrics(yt,yp);
  if scheduler: scheduler.step(vm['macro_f1'])
  entry={'epoch':epoch,'train_loss':total/n,'val_loss':vl,'val_accuracy':vm['accuracy'],'val_macro_f1':vm['macro_f1'],'learning_rate':opt.param_groups[0]['lr'],'elapsed_seconds':time.perf_counter()-start};history.append(entry);print(json.dumps(entry),flush=True)
  with (a.output_dir/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(entry)+'\n');f.flush()
  torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'optimizer_state_dict':opt.state_dict()},a.output_dir/'last.pt')
  improved=vm['macro_f1']>best[0] or (vm['macro_f1']==best[0] and vl<best[1])
  if improved: best=(vm['macro_f1'],vl,epoch);stale=0;torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'val_metrics':vm},a.output_dir/'best_validation.pt')
  else: stale+=1
  if stale>=a.patience: break
 torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'optimizer_state_dict':opt.state_dict()},a.output_dir/'last.pt');(a.output_dir/'history.json').write_text(json.dumps(history,indent=2))
 if a.validation_only:
  (a.output_dir/'metrics.json').write_text(json.dumps({'protocol':'A01-A07 validation-only tuning; A08-A09 not loaded','epochs_completed':epoch,'best_validation_epoch':best[2],'best_validation':best_ck['val_metrics'] if 'best_ck' in locals() else torch.load(a.output_dir/'best_validation.pt',map_location='cpu',weights_only=False)['val_metrics']},indent=2));return
 ck=torch.load(a.output_dir/'best_validation.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state_dict']);model.to(device);xt,yt,_=load_split_data('test',window='central',dataset='cross_subject');tl,ytrue,ypred=evaluate(model,loader(xt,yt,32),device)
 with np.load(_get_processed_dir()/DATASET_ARCHIVES['cross_subject'],allow_pickle=False) as z: subjects=z['subject_ids_test'].copy()
 assert set(subjects)=={'A08','A09'} and len(subjects)==len(ytrue)
 out={'protocol':'A01-A07 development only; A08-A09 tested once after validation selection','device':str(device),'model_parameters':sum(q.numel() for q in model.parameters()),'epochs_completed':epoch,'best_validation_epoch':best[2],'best_validation':ck['val_metrics'],'test_loss':tl,'test':metrics(ytrue,ypred),'test_per_subject':{s:metrics(ytrue[subjects==s],ypred[subjects==s]) for s in ('A08','A09')},'elapsed_seconds':time.perf_counter()-start}
 np.savez_compressed(a.output_dir/'test_predictions.npz',y_true=ytrue,y_pred=ypred,subject_ids=subjects);(a.output_dir/'metrics.json').write_text(json.dumps(out,indent=2));print(json.dumps({'epochs_completed':epoch,'device':str(device)}))
if __name__=='__main__':
 try: main()
 except Exception:
  traceback.print_exc(file=sys.stderr);sys.stderr.flush();raise
