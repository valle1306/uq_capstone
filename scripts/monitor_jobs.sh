#!/bin/bash
# Monitor running UQ experiments on Amarel

echo "=========================================="
echo "🔍 UQ Experiments Monitor"
echo "=========================================="
echo ""

# Check queue status
echo "📋 Job Queue Status:"
squeue -u hpl14 --format="%.18i %.12j %.8T %.10M %.6D %R"
echo ""

# Count running jobs
RUNNING=$(squeue -u hpl14 -t RUNNING | wc -l)
PENDING=$(squeue -u hpl14 -t PENDING | wc -l)
echo "Running: $((RUNNING - 1)), Pending: $((PENDING - 1))"
echo ""

# Check if any jobs exist
if [ $((RUNNING + PENDING)) -le 2 ]; then
    echo "⚠️  No jobs in queue. Have you submitted them?"
    echo "   Run: bash scripts/run_all_experiments.sh"
    exit 0
fi

echo "=========================================="
echo "📁 Output Files Status:"
echo "=========================================="

# Check baseline
if [ -f "runs/baseline/best_model.pth" ]; then
    echo "✅ Baseline: COMPLETED"
else
    if ls runs/baseline/train_*.out 2>/dev/null | grep -q .; then
        LATEST_OUT=$(ls -t runs/baseline/train_*.out | head -1)
        LINES=$(wc -l < "$LATEST_OUT")
        echo "⏳ Baseline: Training... ($LINES log lines)"
        echo "   Latest: tail -f $LATEST_OUT"
    else
        echo "❌ Baseline: Not started"
    fi
fi

# Check MC Dropout
if [ -f "runs/mc_dropout/best_model.pth" ]; then
    echo "✅ MC Dropout: COMPLETED"
else
    if ls runs/mc_dropout/train_*.out 2>/dev/null | grep -q .; then
        LATEST_OUT=$(ls -t runs/mc_dropout/train_*.out | head -1)
        LINES=$(wc -l < "$LATEST_OUT")
        echo "⏳ MC Dropout: Training... ($LINES log lines)"
        echo "   Latest: tail -f $LATEST_OUT"
    else
        echo "❌ MC Dropout: Not started"
    fi
fi

# Check ensemble members
COMPLETED_MEMBERS=0
for i in {0..4}; do
    if [ -f "runs/ensemble/member_$i/best_model.pth" ]; then
        ((COMPLETED_MEMBERS++))
    fi
done

if [ $COMPLETED_MEMBERS -eq 5 ]; then
    echo "✅ Ensemble: COMPLETED (5/5 members)"
elif [ $COMPLETED_MEMBERS -gt 0 ]; then
    echo "⏳ Ensemble: Training... ($COMPLETED_MEMBERS/5 members done)"
    if ls runs/ensemble/member_0/train_*.out 2>/dev/null | grep -q .; then
        LATEST_OUT=$(ls -t runs/ensemble/member_0/train_*.out | head -1)
        echo "   Latest: tail -f $LATEST_OUT"
    fi
else
    echo "❌ Ensemble: Not started"
fi

# Check evaluation
if [ -f "runs/evaluation/results.json" ]; then
    echo "✅ Evaluation: COMPLETED"
    echo ""
    echo "=========================================="
    echo "🎉 ALL EXPERIMENTS COMPLETE!"
    echo "=========================================="
    echo "View results:"
    echo "  cat runs/evaluation/results.json"
    echo "  Display comparison.png"
else
    if ls runs/evaluation/evaluate_*.out 2>/dev/null | grep -q .; then
        echo "⏳ Evaluation: Running..."
        LATEST_OUT=$(ls -t runs/evaluation/evaluate_*.out | head -1)
        echo "   Latest: tail -f $LATEST_OUT"
    else
        echo "❌ Evaluation: Waiting for training to complete"
    fi
fi

echo ""
echo "=========================================="
echo "🔧 Quick Commands:"
echo "=========================================="
echo "Watch baseline:    tail -f runs/baseline/train_*.out"
echo "Watch MC dropout:  tail -f runs/mc_dropout/train_*.out"
echo "Watch ensemble:    tail -f runs/ensemble/member_0/train_*.out"
echo "Check errors:      tail runs/*/train_*.err"
echo "Refresh monitor:   bash scripts/monitor_jobs.sh"
echo ""
