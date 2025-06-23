pip install --upgrade scipy statsmodels

pip freeze > requirements.txt

echo "==== Versions mises à jour ===="
grep -E 'scipy|statsmodels' requirements.txt
echo "==============================="

git add requirements.txt
git commit -m "🔧 MAJ des dépendances : scipy et statsmodels"
git push

echo "✅ Dépendances mises à jour et poussées sur GitHub avec succès."
