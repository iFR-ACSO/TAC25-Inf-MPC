function q_CA = quatMult(q_CB,q_BA)

% just for readability
qvec_CB = q_CB(1:3);
q4_CB   = q_CB(end);

qvec_BA = q_BA(1:3);
q4_BA   = q_BA(end);

% eq. (2.4) Fichter 
q_CA = [q4_CB*eye(3) - cpm(qvec_CB) qvec_CB;
            -qvec_CB'               q4_CB  ]*[qvec_BA; q4_BA];

end