game = [0,2;
        4,1];
        
estimates1 = [0,0];
estimates2 = [0,0];

trials1 = [0,0];
trials2 = [0,0];

epsilon = 0.1;

T = 100000;
for t = 1:T  
  [_, a1] = max(estimates1);
  [_, a2] = min(estimates2);
  if rand()<epsilon
    a1 = randi(2);
  end
  if rand()<epsilon
    a2 = randi(2);
  end
  outcome = game(a1,a2);
  
  estimates1(a1) = (estimates1(a1) * trials1(a1) + outcome) / (trials1(a1) + 1);
  estimates2(a2) = (estimates2(a2) * trials2(a2) + outcome) / (trials2(a2) + 1);
  
  trials1(a1) += 1;
  trials2(a2) += 1;
end
  