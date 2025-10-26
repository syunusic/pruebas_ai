from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('profiles', '0002_create_missing_profiles'),
    ]

    operations = [
        migrations.AddField(
            model_name='alumniprofile',
            name='company_name',
            field=models.CharField(blank=True, default='', max_length=255, verbose_name='empresa'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='alumniprofile',
            name='company_start_year',
            field=models.PositiveIntegerField(
                blank=True,
                help_text='Año en que entraste o fundaste la empresa actual.',
                null=True,
                verbose_name='año de ingreso a la empresa',
            ),
        ),
    ]
