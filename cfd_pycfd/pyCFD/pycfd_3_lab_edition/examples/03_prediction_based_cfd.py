from pycfd import cfd_test_predictions

result = cfd_test_predictions(
    y_true=y_test,
    y_a=domain_a_pred,
    y_b=domain_b_pred,
    y_fusion=fusion_pred,
    name_a="DomainA",
    name_b="DomainB"
)

print(result.summary())
